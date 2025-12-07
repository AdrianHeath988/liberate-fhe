import numpy as np
import torch
import math
from liberate.fhe.presets import types

class BootstrappingContext:
    def __init__(self, engine, verbose=False):
        """
        Args:
            engine: Instance of ckks_engine to access context, rotate, and mult functions.
        """
        self.engine = engine
        self.ctx = engine.ctx
        self.verbose = verbose
        if self.verbose:
            print("init BootstrappingContext with BSGS support")

    def generate_dft_diagonals(self, N, inverse=False):
        """
        Generates the diagonal vectors of the DFT (or IDFT) matrix.
        Returns raw numpy arrays (unencoded).
        """
        # 1. Create the matrix roots
        root = np.exp(-2j * np.pi / (2 * N)) if not inverse else np.exp(2j * np.pi / (2 * N))
        
        # 2. Power calculation for roots matrix[i, j] = root^(i*j)
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        matrix = root ** (i * j)
        
        if inverse:
            matrix /= N

        # 3. Extract diagonals
        # diag[k] is the vector where matrix[i, j] such that j - i = k (mod N)
        diagonals = {}
        for k in range(N):
            d = np.diagonal(matrix, offset=k)
            if len(d) < N:
                d = np.concatenate((d, np.diagonal(matrix, offset=k-N)))
            diagonals[k] = d
            
        return diagonals

    def _mult_ct_encoded_pt(self, ct, pt_tiled, level):
        """
        Helper to multiply a Ciphertext by an already encoded (and tiled) Plaintext.
        Mimics engine.mc_mult logic but skips encoding/scaling of raw messages.
        """
        # 1. Prepare Plaintext: Transform ntt to prepare for multiplication.
        # Note: pt_tiled should already be tiled to the correct level.
        self.engine.ntt.enter_ntt(pt_tiled, level)

        # 2. Prepare Ciphertext copies
        # We clone data to avoid modifying the input 'ct' in place during NTT
        # Accessing data directly: ct.data is (c0, c1)
        # Note: We use engine.clone to be safe, or just clone the list structure if strict performance is needed
        new_ct = self.engine.clone(ct)
        
        self.engine.ntt.enter_ntt(new_ct.data[0], level)
        self.engine.ntt.enter_ntt(new_ct.data[1], level)

        # 3. Montgomery Multiplication
        new_d0 = self.engine.ntt.mont_mult(pt_tiled, new_ct.data[0], level)
        new_d1 = self.engine.ntt.mont_mult(pt_tiled, new_ct.data[1], level)

        # 4. Inverse NTT and Reduce
        self.engine.ntt.intt_exit_reduce(new_d0, level)
        self.engine.ntt.intt_exit_reduce(new_d1, level)

        # 5. Update Data
        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        # 6. Rescale
        return self.engine.rescale(new_ct)

    def bsgs_linear_transform(self, ct, diagonals, galk, level):
        """
        Performs Matrix-Vector multiplication using Baby-Step Giant-Step algorithm.
        """
        N = self.ctx.N // 2 # Number of slots
        n1 = int(math.ceil(math.sqrt(N))) # Giant step dimension
        n2 = int(math.ceil(N / n1))       # Baby step dimension

        # --- Pre-processing Diagonals ---
        # We must scale the raw diagonals by sqrt(deviation) BEFORE encoding,
        # just like engine.mc_mult does.
        scale_factor = np.sqrt(self.engine.deviations[level + 1])
        
        encoded_diagonals = {}
        for k, diag in diagonals.items():
            # 1. Scale
            scaled_diag = diag * scale_factor
            
            # 2. Encode (returns list of tensors)
            # We encode at level 0 usually, tiling handles the rest
            encoded_pt = self.engine.encode(scaled_diag, level=0)
            
            # 3. Tile immediately to the target level to save time in the loop
            # This prepares the PT for the specific level of the ciphertext
            tiled_pt = self.engine.ntt.tile_unsigned(encoded_pt, level)
            encoded_diagonals[k] = tiled_pt

        # --- BSGS Execution ---
        
        # 1. Compute inner rotations (Giant steps)
        # We need rotations of ct by g*i for i in 0..n1
        temp_rotations = {}
        for i in range(n1):
            rotation_idx = n1 * i
            if rotation_idx == 0:
                temp_rotations[rotation_idx] = ct
            else:
                # Rotate
                temp_rotations[rotation_idx] = self.engine.rotate_galois(ct, galk, rotation_idx)

        final_sum = None

        # 2. Accumulate
        for j in range(n2):
            inner_sum = None
            
            for i in range(n1):
                rot_idx = n1 * i
                current_diag_idx = (rot_idx + j) % N
                
                # Retrieve pre-rotated ciphertext
                rot_ct = temp_rotations[rot_idx]
                
                # Retrieve pre-encoded, pre-tiled plaintext
                pt_tiled = encoded_diagonals[current_diag_idx]
                
                # Multiply: diag * rot(ct)
                # USE CUSTOM HELPER instead of engine.cm_mult
                term = self._mult_ct_encoded_pt(rot_ct, pt_tiled, level)
                
                if inner_sum is None:
                    inner_sum = term
                else:
                    inner_sum = self.engine.add(inner_sum, term)
            
            # 3. Outer rotation (Baby step) by j
            if j != 0:
                inner_sum = self.engine.rotate_galois(inner_sum, galk, j)
            
            if final_sum is None:
                final_sum = inner_sum
            else:
                final_sum = self.engine.add(final_sum, inner_sum)

        return final_sum

    def modup(self, ct, target_level=0):
        """
        Modulus Raising (ModUp).
        Transitions ciphertext from current modulus q to a larger modulus Q (Level 0).
        """
        print(f"[BootstrappingContext] Starting ModUp from level {ct.level} to {target_level}...")
        
        # 1. Transform to Coefficient Form (INTT)
        ct_coeff = self.engine.clone(ct)
        self.engine.ntt.intt_exit_reduce(ct_coeff.data[0], ct_coeff.level)
        self.engine.ntt.intt_exit_reduce(ct_coeff.data[1], ct_coeff.level)

        # 2. Basis Extension (Lifting to Level 0)
        num_devices = self.engine.ntt.num_devices
        acc0 = [None] * num_devices
        acc1 = [None] * num_devices

        # Helper to align single tensor to device list
        def align(tensor, dev_idx):
            l = []
            for i in range(num_devices):
                if i == dev_idx:
                    l.append(tensor)
                else:
                    # Create empty tensor with 0 elements but valid dimension for safety
                    # Check dimension to match packed_accessor requirements (1D for scalars, 2D for coeffs)
                    if tensor.dim() == 1:
                        l.append(torch.empty(0, device=self.engine.ntt.devices[i], dtype=tensor.dtype))
                    else:
                        l.append(torch.empty(0, 0, device=self.engine.ntt.devices[i], dtype=tensor.dtype))
            return l

        # Iterate over all source devices
        for src_device in range(num_devices):
            parts = self.engine.ntt.p.p[ct.level][src_device]
            
            for part_id in range(len(parts)):
                # A. Pre-computation
                state0 = self.engine.pre_extend(ct_coeff.data[0], src_device, ct.level, part_id, exit_ntt=False)
                state1 = self.engine.pre_extend(ct_coeff.data[1], src_device, ct.level, part_id, exit_ntt=False)
                
                part_range = tuple(parts[part_id])
                pack = self.engine.ntt.parts_pack[src_device][part_range]
                
                # B. Extend to all target devices
                for dst_device in range(num_devices):
                    if src_device != dst_device:
                        s0 = state0.to(self.engine.ntt.devices[dst_device])
                        s1 = state1.to(self.engine.ntt.devices[dst_device])
                    else:
                        s0 = state0
                        s1 = state1
                    
                    # --- Custom Extension Logic ---
                    rns_len = len(self.engine.ntt.p.destination_arrays_with_special[target_level][dst_device])
                    
                    # Replicate state
                    ext0 = s0[0].repeat(rns_len, 1)
                    ext1 = s1[0].repeat(rns_len, 1)
                    
                    # Enter Montgomery at TARGET level
                    # Retrieve the correct Rs (R^2) tensor for this device/level/mult_type
                    # Rs_prepack[dst][lvl][-2] is a LIST. We take the single tensor inside.
                    rs_list = self.engine.ntt.Rs_prepack[dst_device][target_level][-2]
                    rs_tensor = rs_list[0]
                    
                    # Align inputs to device lists
                    ext0_aligned = align(ext0, dst_device)
                    ext1_aligned = align(ext1, dst_device)
                    rs_aligned = align(rs_tensor, dst_device)

                    self.engine.ntt.mont_enter_scalar(ext0_aligned, rs_aligned, target_level, dst_device, -2)
                    self.engine.ntt.mont_enter_scalar(ext1_aligned, rs_aligned, target_level, dst_device, -2)
                    
                    # Add Corrections (L_enter)
                    L_enter_list = pack['L_enter'][dst_device]
                    
                    if L_enter_list is not None:
                         alpha = len(s0)
                         start = self.engine.ntt.starts[target_level][dst_device]
                         
                         for i in range(alpha - 1):
                             Y0 = s0[i+1].repeat(rns_len, 1)
                             Y1 = s1[i+1].repeat(rns_len, 1)
                             
                             Y0_aligned = align(Y0, dst_device)
                             Y1_aligned = align(Y1, dst_device)
                             
                             # Extract and align correction tensor
                             Li_tensor = L_enter_list[i][start:]
                             Li_aligned = align(Li_tensor, dst_device)
                             
                             self.engine.ntt.mont_enter_scalar(Y0_aligned, Li_aligned, target_level, dst_device, -2)
                             self.engine.ntt.mont_enter_scalar(Y1_aligned, Li_aligned, target_level, dst_device, -2)
                             
                             res0 = self.engine.ntt.mont_add(ext0_aligned, Y0_aligned, target_level, dst_device, -2)
                             res1 = self.engine.ntt.mont_add(ext1_aligned, Y1_aligned, target_level, dst_device, -2)
                             
                             ext0_aligned = res0
                             ext1_aligned = res1
                             
                             ext0 = ext0_aligned[dst_device]
                             ext1 = ext1_aligned[dst_device]

                    # Accumulate
                    if acc0[dst_device] is None:
                        acc0[dst_device] = ext0
                        acc1[dst_device] = ext1
                    else:
                        # Re-align because acc is just a tensor
                        ext0_aligned = align(ext0, dst_device)
                        ext1_aligned = align(ext1, dst_device)
                        
                        acc0_aligned = align(acc0[dst_device], dst_device)
                        acc1_aligned = align(acc1[dst_device], dst_device)

                        res0 = self.engine.ntt.mont_add(acc0_aligned, ext0_aligned, target_level, dst_device, -2)
                        res1 = self.engine.ntt.mont_add(acc1_aligned, ext1_aligned, target_level, dst_device, -2)
                        
                        acc0[dst_device] = res0[dst_device]
                        acc1[dst_device] = res1[dst_device]

        # 3. Finalize
        new_ct0_list = []
        new_ct1_list = []
        
        for device_id in range(num_devices):
            if acc0[device_id] is None:
                raise RuntimeError(f"Device {device_id} received no data during ModUp.")
            new_ct0_list.append(acc0[device_id])
            new_ct1_list.append(acc1[device_id])

        self.engine.ntt.ntt(new_ct0_list, target_level, mult_type=-2)
        self.engine.ntt.ntt(new_ct1_list, target_level, mult_type=-2)

        extended_ct = ct_coeff._replace(
            data=(new_ct0_list, new_ct1_list),
            level=target_level,
            ntt_state=True,
            montgomery_state=True,
            include_special=True
        )

        return extended_ct

    def ctos(self, ct, galk):
        """
        Ciphertext-to-Slot (Homomorphic Decode).
        """
        print("[BootstrappingContext] Starting CTOS (BSGS)...")
        
        # 1. Generate Diagonals for Inverse DFT
        N = self.ctx.N // 2
        diags = self.generate_dft_diagonals(N, inverse=True)
        
        # 2. Apply Linear Transform
        result = self.bsgs_linear_transform(ct, diags, galk, ct.level)
        
        return result

    def stoc(self, ct, galk):
        """
        Slot-to-Coeff (Homomorphic Encode).
        Performs the forward DFT to pack coefficients back into the canonical embedding slots.
        """
        print("[BootstrappingContext] Starting STOC (BSGS)...")
        
        # 1. Generate Diagonals for Forward DFT
        # We use inverse=False to get the standard DFT matrix
        N = self.ctx.N // 2
        diags = self.generate_dft_diagonals(N, inverse=False)
        
        # 2. Apply Linear Transform using the existing BSGS infrastructure
        # The linear transform logic is symmetric for any square matrix (DFT/IDFT)
        result = self.bsgs_linear_transform(ct, diags, galk, ct.level)
        
        return result

    def eval_taylor_mod(self, ct, degree, evk, q_boot=None):
        """
        Homomorphic Modular Reduction using Taylor Expansion of Sine.
        Approximates f(x) = (q/2pi) * sin(2pi * x / q) to remove q*I error.
        
        Args:
            ct: The ciphertext (usually output of CTOS).
            degree: The degree of the Taylor expansion (should be odd).
            evk: Evaluation key (needed for multiplication).
            q_boot: The modulus q corresponding to the bootstrapping level (period).
                    If None, defaults to the engine's scale (assuming scale ~ q).
        """
        print(f"[BootstrappingContext] Starting Taylor EvalMod (degree={degree})...")
        
        if q_boot is None:
            q_boot = self.engine.ctx.q[self.engine.num_levels - 1]

        # Pre-calculate constants
        two_pi_over_q = 2 * math.pi / q_boot
        
        # Taylor series for (q/2pi) * sin(u) where u = (2pi/q) * x
        # coeff_k = (q/2pi) * (-1)^k / (2k+1)! * (2pi/q)^(2k+1)
        #         = (-1)^k / (2k+1)! * (2pi/q)^(2k)
        
        # We accumulate the result: result = sum(coeff_i * x^i)
        # Optimized: calculate powers of x iteratively to save levels if possible,
        # but for simplicity and utilizing engine features, we compute powers directly.
        
        # 1. First term (k=0): x
        # coeff = 1.0
        result = ct # Copy not strictly needed if we don't modify in place immediately, but engine operations usually return new
        
        # We need x^2 to step up powers (x, x^3, x^5...)
        if degree >= 3:
            # Calculate x^2
            x2 = self.engine.square(ct, evk)
            
            # Current power x^p starts at x^1
            current_pow = ct
            
            # Loop for k = 1, 2, ...
            # Terms: x^3, x^5, ...
            for d in range(3, degree + 1, 2):
                k = (d - 1) // 2
                
                # Update power: x^d = x^{d-2} * x^2
                current_pow = self.engine.mult(current_pow, x2, evk)
                
                # Calculate coefficient
                # coeff = (-1)^k * (1/(2k+1)!) * (2pi/q)^(2k)
                factorial_val = math.factorial(d)
                ratio = (two_pi_over_q ** (d - 1)) # Note: formula simplification resulted in (2pi/q)^(2k)
                
                coeff = ((-1)**k) * (1.0 / factorial_val) * ratio
                
                # Add term: coeff * x^d
                term = self.engine.mult_scalar(current_pow, coeff)
                result = self.engine.add(result, term)
                
        return result