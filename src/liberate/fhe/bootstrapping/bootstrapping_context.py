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