import numpy as np
import torch
import math
import torch.cuda.nccl as nccl
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
            
        # Create persistent streams for multi-device operations.
        # These are used for side devices to avoid legacy stream errors during capture.
        self.comm_streams = []
        for d in self.engine.ntt.devices:
            with torch.cuda.device(d):
                self.comm_streams.append(torch.cuda.Stream(device=d))

    def generate_dft_diagonals(self, N, inverse=False):
        """
        Generates the diagonal vectors of the DFT (or IDFT) matrix.
        Returns raw numpy arrays (unencoded).
        """
        root = np.exp(-2j * np.pi / (2 * N)) if not inverse else np.exp(2j * np.pi / (2 * N))
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        matrix = root ** (i * j)
        if inverse:
            matrix /= N

        diagonals = {}
        for k in range(N):
            d = np.diagonal(matrix, offset=k)
            if len(d) < N:
                d = np.concatenate((d, np.diagonal(matrix, offset=k-N)))
            diagonals[k] = d
            
        return diagonals

    def _mult_ct_encoded_pt(self, ct, pt_tiled, level):
        self.engine.ntt.enter_ntt(pt_tiled, level)
        new_ct = self.engine.clone(ct)
        
        self.engine.ntt.enter_ntt(new_ct.data[0], level)
        self.engine.ntt.enter_ntt(new_ct.data[1], level)

        new_d0 = self.engine.ntt.mont_mult(pt_tiled, new_ct.data[0], level)
        new_d1 = self.engine.ntt.mont_mult(pt_tiled, new_ct.data[1], level)

        self.engine.ntt.intt_exit_reduce(new_d0, level)
        self.engine.ntt.intt_exit_reduce(new_d1, level)

        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        return self.engine.rescale(new_ct)

    def bsgs_linear_transform(self, ct, diagonals, galk, level):
        N = self.ctx.N // 2 
        n1 = int(math.ceil(math.sqrt(N))) 
        n2 = int(math.ceil(N / n1))       

        scale_factor = np.sqrt(self.engine.deviations[level + 1])
        encoded_diagonals = {}
        for k, diag in diagonals.items():
            scaled_diag = diag * scale_factor
            encoded_pt = self.engine.encode(scaled_diag, level=0)
            tiled_pt = self.engine.ntt.tile_unsigned(encoded_pt, level)
            encoded_diagonals[k] = tiled_pt

        temp_rotations = {}
        for i in range(n1):
            rotation_idx = n1 * i
            if rotation_idx == 0:
                temp_rotations[rotation_idx] = ct
            else:
                temp_rotations[rotation_idx] = self.engine.rotate_galois(ct, galk, rotation_idx)

        final_sum = None

        for j in range(n2):
            inner_sum = None
            for i in range(n1):
                rot_idx = n1 * i
                current_diag_idx = (rot_idx + j) % N
                
                rot_ct = temp_rotations[rot_idx]
                pt_tiled = encoded_diagonals[current_diag_idx]
                
                term = self._mult_ct_encoded_pt(rot_ct, pt_tiled, level)
                
                if inner_sum is None:
                    inner_sum = term
                else:
                    inner_sum = self.engine.add(inner_sum, term)
            
            if j != 0:
                inner_sum = self.engine.rotate_galois(inner_sum, galk, j)
            
            if final_sum is None:
                final_sum = inner_sum
            else:
                final_sum = self.engine.add(final_sum, inner_sum)

        return final_sum

    def modup(self, ct, target_level=0):
        import torch.cuda.nccl as nccl
        
        if self.verbose:
            print(f"[BootstrappingContext] Starting ModUp from level {ct.level} to {target_level}...")
        
        # 1. Transform to Coefficient Form (INTT)
        ct_coeff = self.engine.clone(ct)
        self.engine.ntt.intt_exit_reduce(ct_coeff.data[0], ct_coeff.level)
        self.engine.ntt.intt_exit_reduce(ct_coeff.data[1], ct_coeff.level)

        num_devices = self.engine.ntt.num_devices
        acc0 = [None] * num_devices
        acc1 = [None] * num_devices
        
        # Access persistent streams from engine
        comm_streams = self.engine.comm_streams

        # Sync Default -> Comm
        for i in range(num_devices):
             with torch.cuda.device(self.engine.ntt.devices[i]):
                 comm_streams[i].wait_stream(torch.cuda.current_stream())

        def align(tensor, dev_idx):
            return [tensor]

        # Iterate over source devices
        for src_device in range(num_devices):
            parts = self.engine.ntt.p.p[ct.level][src_device]
            if len(parts) == 0:
                continue

            # Switch to source comm stream
            with torch.cuda.device(self.engine.ntt.devices[src_device]), torch.cuda.stream(comm_streams[src_device]):
                src_tensors = []
                meta_map = [] 
                for part_id in range(len(parts)):
                    s0 = self.engine.pre_extend(ct_coeff.data[0], src_device, ct.level, part_id, exit_ntt=False)
                    s1 = self.engine.pre_extend(ct_coeff.data[1], src_device, ct.level, part_id, exit_ntt=False)
                    src_tensors.append(s0)
                    src_tensors.append(s1)
                    meta_map.append((part_id, 0))
                    meta_map.append((part_id, 1))

                split_sizes = [t.size(0) for t in src_tensors]
                stacked_src = torch.cat(src_tensors, dim=0)

            # NCCL Broadcast
            nccl_list = [None] * num_devices
            nccl_list[src_device] = stacked_src
            
            for dst_dev in range(num_devices):
                if dst_dev != src_device:
                    with torch.cuda.device(self.engine.ntt.devices[dst_dev]), torch.cuda.stream(comm_streams[dst_dev]):
                        nccl_list[dst_dev] = torch.empty_like(stacked_src, device=self.engine.ntt.devices[dst_dev])

            nccl.broadcast(nccl_list, root=src_device, streams=comm_streams)

            # Process on Destinations
            for dst_device in range(num_devices):
                with torch.cuda.device(self.engine.ntt.devices[dst_device]), torch.cuda.stream(comm_streams[dst_device]):
                    received_stack = nccl_list[dst_device]
                    chunks = torch.split(received_stack, split_sizes, dim=0)
                    
                    for i in range(0, len(chunks), 2):
                        s0 = chunks[i]
                        s1 = chunks[i+1]
                        part_id = meta_map[i][0]
                        
                        rns_len = len(self.engine.ntt.p.destination_arrays_with_special[target_level][dst_device])
                        ext0 = s0[0].repeat(rns_len, 1)
                        ext1 = s1[0].repeat(rns_len, 1)
                        
                        rs_list = self.engine.ntt.Rs_prepack[dst_device][target_level][-2]
                        rs_tensor = rs_list[0]
                        
                        ext0_aligned = align(ext0, dst_device)
                        ext1_aligned = align(ext1, dst_device)
                        rs_aligned = align(rs_tensor, dst_device)

                        self.engine.ntt.mont_enter_scalar(ext0_aligned, rs_aligned, target_level, dst_device, -2)
                        self.engine.ntt.mont_enter_scalar(ext1_aligned, rs_aligned, target_level, dst_device, -2)
                        
                        part_range = tuple(parts[part_id])
                        pack = self.engine.ntt.parts_pack[src_device][part_range]
                        L_enter_list = pack['L_enter'][dst_device]
                        
                        if L_enter_list is not None:
                             alpha = len(s0)
                             start = self.engine.ntt.starts[target_level][dst_device]
                             
                             for k in range(alpha - 1):
                                 Y0 = s0[k+1].repeat(rns_len, 1)
                                 Y1 = s1[k+1].repeat(rns_len, 1)
                                 Y0_aligned = align(Y0, dst_device)
                                 Y1_aligned = align(Y1, dst_device)
                                 Li_tensor = L_enter_list[k][start:]
                                 Li_aligned = align(Li_tensor, dst_device)
                                 
                                 self.engine.ntt.mont_enter_scalar(Y0_aligned, Li_aligned, target_level, dst_device, -2)
                                 self.engine.ntt.mont_enter_scalar(Y1_aligned, Li_aligned, target_level, dst_device, -2)
                                 res0 = self.engine.ntt.mont_add(ext0_aligned, Y0_aligned, target_level, dst_device, -2)
                                 res1 = self.engine.ntt.mont_add(ext1_aligned, Y1_aligned, target_level, dst_device, -2)
                                 ext0_aligned = res0
                                 ext1_aligned = res1
                                 ext0 = ext0_aligned[0]
                                 ext1 = ext1_aligned[0]

                        if acc0[dst_device] is None:
                            acc0[dst_device] = ext0
                            acc1[dst_device] = ext1
                        else:
                            ext0_aligned = align(ext0, dst_device)
                            ext1_aligned = align(ext1, dst_device)
                            acc0_aligned = align(acc0[dst_device], dst_device)
                            acc1_aligned = align(acc1[dst_device], dst_device)

                            res0 = self.engine.ntt.mont_add(acc0_aligned, ext0_aligned, target_level, dst_device, -2)
                            res1 = self.engine.ntt.mont_add(acc1_aligned, ext1_aligned, target_level, dst_device, -2)
                            
                            acc0[dst_device] = res0[0]
                            acc1[dst_device] = res1[0]

        # Sync Comm -> Default
        for i in range(num_devices):
             with torch.cuda.device(self.engine.ntt.devices[i]):
                 torch.cuda.current_stream().wait_stream(comm_streams[i])

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
        print("[BootstrappingContext] Starting CTOS (BSGS)...")
        N = self.ctx.N // 2
        diags = self.generate_dft_diagonals(N, inverse=True)
        result = self.bsgs_linear_transform(ct, diags, galk, ct.level)
        return result

    def stoc(self, ct, galk):
        print("[BootstrappingContext] Starting STOC (BSGS)...")
        N = self.ctx.N // 2
        diags = self.generate_dft_diagonals(N, inverse=False)
        result = self.bsgs_linear_transform(ct, diags, galk, ct.level)
        return result

    def eval_taylor_mod(self, ct, degree, evk, q_boot=None):
        print(f"[BootstrappingContext] Starting Taylor EvalMod (degree={degree})...")
        if q_boot is None:
            q_boot = self.engine.ctx.q[self.engine.num_levels - 1]

        two_pi_over_q = 2 * math.pi / q_boot
        result = ct 
        
        if degree >= 3:
            x2 = self.engine.square(ct, evk)
            current_pow = ct
            for d in range(3, degree + 1, 2):
                k = (d - 1) // 2
                current_pow = self.engine.mult(current_pow, x2, evk)
                
                factorial_val = math.factorial(d)
                ratio = (two_pi_over_q ** (d - 1)) 
                
                coeff = ((-1)**k) * (1.0 / factorial_val) * ratio
                
                term = self.engine.mult_scalar(current_pow, coeff)
                result = self.engine.add(result, term)
                
        return result