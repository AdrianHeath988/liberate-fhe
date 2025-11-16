
import numpy as np
from liberate.ntt import ntt_context
from liberate.fhe import bootstrapping as cuda_boot

class BootstrappingContext:
    def __init__(self, ctx, devices=None, verbose=False):
        # reuse ntt_context for NTT parameter packing and device tensors
        self.ntt = ntt_context(ctx, devices=devices, verbose=verbose)
        self.ctx = ctx
        self.devices = self.ntt.devices

    @staticmethod
    def make_moduli_array(moduli_iterable):
        dt = np.dtype([('p','u8'),('p_twice','u8'),('p_word_size','u8'),('p_mod_inv','u8')])
        arr = np.zeros(len(moduli_iterable), dtype=dt)
        for i, m in enumerate(moduli_iterable):
            p, p2, pw, inv = m
            arr['p'][i] = p
            arr['p_twice'][i] = p2
            arr['p_word_size'][i] = pw
            arr['p_mod_inv'][i] = inv
        return arr

    def ctos(self, ct_in: np.ndarray, moduli, n_power:int, q_size:int, p_size:int):
        ct_in = np.ascontiguousarray(ct_in, dtype=np.uint64)
        if not hasattr(moduli, 'dtype'):
            moduli = self.make_moduli_array(moduli)
        # cuda_boot.ctos_gpu is the binding registered under liberate_fhe_cuda.bootstrapping
        return cuda_boot.ctos_gpu(ct_in, moduli, n_power, q_size, p_size)

    def mod_raise(self, ct_in, ntt_table, intt_table, moduli, n_power, q_size, p_size):
        ct_in = np.ascontiguousarray(ct_in, dtype=np.uint64)
        ntt_table = np.ascontiguousarray(ntt_table, dtype=np.uint64)
        intt_table = np.ascontiguousarray(intt_table, dtype=np.uint64)
        if not hasattr(moduli, 'dtype'):
            moduli = self.make_moduli_array(moduli)
        return cuda_boot.mod_raise_gpu(ct_in, ntt_table, intt_table, moduli, n_power, q_size, p_size)