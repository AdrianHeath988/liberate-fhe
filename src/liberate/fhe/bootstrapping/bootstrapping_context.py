
import numpy as np
from liberate.ntt import ntt_context
from liberate.liberate_fhe_cuda import bootstrapping as cuda_boot

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
        print("In BootstrappingContext ctos, going even deeper")
        return cuda_boot.ctos_gpu(ct_in, moduli, n_power, q_size, p_size)

    def mod_raise(self, ct_in, ntt_table, intt_table, moduli, n_power, q_size, p_size):
        ct_in = np.ascontiguousarray(ct_in, dtype=np.uint64)
        ntt_table = np.ascontiguousarray(ntt_table, dtype=np.uint64)
        intt_table = np.ascontiguousarray(intt_table, dtype=np.uint64)
        return cuda_boot.mod_raise_gpu(ct_in, ntt_table, intt_table, moduli, n_power, q_size, p_size)