
import numpy as np
from liberate.ntt import ntt_context
from liberate.liberate_fhe_cuda import bootstrapping as cuda_boot
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
    def __init__(self, ctx, devices=None, verbose=False):
        # reuse ntt_context for NTT parameter packing and device tensors
        # self.ntt = ntt_context(ctx, devices=devices, verbose=verbose)
        # self.ctx = ctx
        # self.devices = self.ntt.devices
        print("init BootstrappingContext")

    def ctos(self, ct_in: np.ndarray, moduli, n_power:int, q_size:int, p_size:int):
        #ct_in = np.ascontiguousarray(ct_in, dtype=np.uint64)
        # cuda_boot.ctos_gpu is the binding registered under liberate_fhe_cuda.bootstrapping
        ct_np, moduli_np = _ct_to_numpy_uint64(ct_in, None) # 2nd should be moduli
        print (moduli_np)
        print (ct_np)
        print("In BootstrappingContext ctos, going even deeper")
        return cuda_boot.ctos_gpu(ct_np, moduli_np, int(n_power), int(q_size), int(p_size))

    def mod_raise(self, ct_in, ntt_table, intt_table, moduli, n_power, q_size, p_size):
        ct_in = np.ascontiguousarray(ct_in, dtype=np.uint64)
        ntt_table = np.ascontiguousarray(ntt_table, dtype=np.uint64)
        intt_table = np.ascontiguousarray(intt_table, dtype=np.uint64)
        return cuda_boot.mod_raise_gpu(ct_in, ntt_table, intt_table, moduli, n_power, q_size, p_size)