
import numpy as np
from liberate.ntt import ntt_context
from liberate.liberate_fhe_cuda import bootstrapping as cuda_boot
from liberate.fhe.presets import types
def _ct_to_numpy_uint64(ct_in, moduli=None):
    """Flatten ct_in.data (list-of-lists of torch tensors) into a numpy.uint64 array.
    Also produce a NumPy array compatible with the C `Modulus64` struct.

    Returns a tuple: (ct_np, moduli_np)

    If `moduli` is provided (array-like), it is converted to a suitable NumPy
    representation. If `moduli` is None, a single-row placeholder structured
    array of zeros is returned so the native call has a valid moduli buffer.
    """
    parts = []
    for depth in ct_in.data:            # depth: list of torch.Tensor
        for t in depth:
            # ensure tensor is on CPU and convert to uint64 numpy
            parts.append(t.detach().cpu().numpy().astype(np.uint64))
    # stack/reshape depending on expected layout; try a 2D array (components x coeffs)
    ct_np = np.vstack(parts)

    # Prepare moduli buffer compatible with struct Modulus64 { uint64_t p; uint64_t p_twice; uint64_t p_word_size; uint64_t p_mod_inv; }
    if moduli is None:
        # Create a single zeroed placeholder row (dtype itemsize 32 bytes)
        dt = np.dtype([('p', 'u8'), ('p_twice', 'u8'), ('p_word_size', 'u8'), ('p_mod_inv', 'u8')])
        mod_np = np.zeros(1, dtype=dt)
    else:
        # If caller passed a flat (N,4) uint64 array or an iterable, normalize it.
        arr = np.asarray(moduli)
        if arr.ndim == 2 and arr.shape[1] == 4 and arr.dtype == np.uint64:
            # convert (N,4) -> structured dtype
            dt = np.dtype([('p', 'u8'), ('p_twice', 'u8'), ('p_word_size', 'u8'), ('p_mod_inv', 'u8')])
            mod_np = np.zeros(arr.shape[0], dtype=dt)
            mod_np['p'] = arr[:, 0]
            mod_np['p_twice'] = arr[:, 1]
            mod_np['p_word_size'] = arr[:, 2]
            mod_np['p_mod_inv'] = arr[:, 3]
        elif hasattr(moduli, 'dtype') and isinstance(moduli, np.ndarray) and moduli.dtype.itemsize == 32:
            # assume already a compatible structured array
            mod_np = moduli.copy()
        else:
            # Try to convert an iterable of tuples (p, p_twice, p_word_size, p_mod_inv)
            try:
                seq = list(moduli)
                dt = np.dtype([('p', 'u8'), ('p_twice', 'u8'), ('p_word_size', 'u8'), ('p_mod_inv', 'u8')])
                mod_np = np.zeros(len(seq), dtype=dt)
                for i, tpl in enumerate(seq):
                    mod_np['p'][i] = np.uint64(tpl[0])
                    mod_np['p_twice'][i] = np.uint64(tpl[1])
                    mod_np['p_word_size'][i] = np.uint64(tpl[2])
                    mod_np['p_mod_inv'][i] = np.uint64(tpl[3])
            except Exception:
                raise ValueError("Unsupported moduli format; provide structured dtype, (N,4) uint64 array, or iterable of 4-tuples")

    return ct_np, mod_np

class BootstrappingContext:
    def __init__(self, engine, ctx, devices=None, verbose=False):
        self.engine = engine  # Store the engine to access .rotate_single, .mult, .add
        self.ctx = ctx
        self.devices = devices
        
        # Pre-compute standard parameters
        self.N = ctx.N
        self.slots = self.N // 2
        # BSGS Parameters (e.g., sqrt(slots))
        self.baby_steps = int(np.ceil(np.sqrt(self.slots)))
        self.giant_steps = int(np.ceil(self.slots / self.baby_steps))

    def generate_diagonal(self, k, level):
        """
        Generates the k-th diagonal of the DFT matrix.
        TODO: replace this with a CUDA Kernel later
        """
        # 1. Generate roots of unity U[i] = zeta^i
        # 2. Create diagonal vector: d[i] = U[i * k] (Simplified)
        # 3. Encode this vector to a Plaintext (pt) using engine.encode
        #BUT: must ensure it is encoded to the correct 'level'.
        
        # Placeholder logic:
        m_vec = np.array([np.exp(2j * np.pi * i * k / (2*self.slots)) for i in range(self.slots)])
        return self.engine.encode(m_vec, level=level)

    def ctos(self, ct_in):
        """
        Performs CoeffToSlot using BSGS in Python.
        """
        current_level = ct_in.level
        
        #Giant Step Loop
        total_sum = None
        
        for g in range(self.giant_steps):
            giant_rot_idx = g * self.baby_steps
            
            # Inner Accumulator
            inner_sum = None
            
            # Baby Step Loop
            for b in range(self.baby_steps):
                rot_idx = giant_rot_idx + b
                if rot_idx >= self.slots: 
                    break

                # 1. Get the Rotation Key
                # Ensure you have generated these keys in your setup phase!
                rotk_name = f"{types.origins['rotk']}{rot_idx}"
                # You might need a lookup dictionary for keys passed in args
                # rotk = self.engine.galois_keys[rot_idx]
                
                # 2. Rotate Ciphertext (Reuses KeySwitch/ModRaise from engine)
                # Note: To optimize, rotate 'ct_in' only once for the baby step 
                # if the math allows, or rotate the accumulated result. 
                # Standard BSGS: inner_sum += diag_k * rot_k(ct)
                
                # Perform Rotation
                # rotated_ct = self.engine.rotate_single(ct_in, rotk)
                
                # 3. Get Diagonal (Plaintext)
                # diag_pt = self.generate_diagonal(rot_idx, current_level)
                
                # 4. Multiply
                # term = self.engine.mult(rotated_ct, diag_pt, relin=False)
                
                # 5. Accumulate
                # if inner_sum is None: inner_sum = term
                # else: inner_sum = self.engine.add(inner_sum, term)

            # --- Apply Giant Rotation to the inner sum ---
            # This is the optimization: Rotate the SUM, not every term.
            # giant_rotk = ...
            # rotated_inner = self.engine.rotate_single(inner_sum, giant_rotk)
            
            # if total_sum is None: total_sum = rotated_inner
            # else: total_sum = self.engine.add(total_sum, rotated_inner)
            
        return total_sum