import numpy as np
import torch
import contextlib
from liberate.fhe.ckks_engine import ckks_engine

def copy_data_struct(src, dst):
    """Helper to copy data into the graph's fixed memory addresses."""
    def recursive_copy(src_data, dst_data):
        if isinstance(src_data, torch.Tensor):
            dst_data.copy_(src_data)
        elif isinstance(src_data, (list, tuple)):
            for s, d in zip(src_data, dst_data):
                recursive_copy(s, d)
    recursive_copy(src.data, dst.data)

def test_ctos():
    print("--- Setting up CKKS Context ---")
    
    ctx_params = {
        "buffer_bit_length": 62,
        "scale_bits": 40,
        "logN": 15,
        "num_scales": None,
        "num_special_primes": 2,
        "sigma": 3.2,
        "security_bits": 128,
        "quantum": "post_quantum",
        'distribution': "uniform",
        "read_cache": True,
        "save_cache": True,
        "verbose": False, 
        "devices": [0, 1, 2],
    }
    
    engine = ckks_engine(**ctx_params)
    
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(sk=secret_key)
    evk = engine.create_evk(secret_key)
    galk = engine.create_bootstrapping_keys(sk=secret_key)

    depleted_level = engine.num_levels - 1
    test_message = engine.example(amin=-1, amax=1)
    pt = engine.encode(m=test_message, level=depleted_level)
    ct_template = engine.encrypt(pt=pt, pk=public_key, level=depleted_level)

    print("--- 1. Warmup (Standard Execution) ---")
    engine.bootstrap(ct=ct_template, galk=galk, evk=evk)
    torch.cuda.synchronize()
    print("   Warmup complete.")

    # ---------------------------------------------------------
    # STEP 2: CAPTURE CUDA GRAPH (Multi-Device)
    # ---------------------------------------------------------
    print("--- 2. Capturing CUDA Graph ---")
    
    static_input_ct = engine.clone(ct_template)
    static_output_ct = None
    
    graphs = []
    
    # 1. Begin Capture on All Streams
    #    - Device 0: Default Stream
    #    - Device 1+: Comm Stream
    
    # We use ExitStack to manage the stream contexts during bootstrap execution
    with contextlib.ExitStack() as stack:
        
        # A. Setup Capture Contexts
        for i, device_id in enumerate(engine.ntt.devices):
            if i == 0:
                # Device 0: Capture on Default Stream
                s = torch.cuda.default_stream(device_id)
            else:
                # Side Devices: Capture on Persistent Comm Stream
                s = engine.comm_streams[i]
            
            # Switch to this device and stream
            stack.enter_context(torch.cuda.device(device_id))
            stack.enter_context(torch.cuda.stream(s))
            
            # Create and start graph
            g = torch.cuda.CUDAGraph()
            g.capture_begin()
            graphs.append(g)

        # B. Execute Function (Now fully captured on all devices)
        print("   Recording kernels...")
        static_output_ct = engine.bootstrap(ct=static_input_ct, galk=galk, evk=evk)

        # C. End Capture
        # We must end capture in the same order/context
        for i, g in enumerate(graphs):
             device_id = engine.ntt.devices[i]
             if i == 0:
                 s = torch.cuda.default_stream(device_id)
             else:
                 s = engine.comm_streams[i]
                 
             with torch.cuda.device(device_id), torch.cuda.stream(s):
                 g.capture_end()

    print("   Capture complete.")

    # ---------------------------------------------------------
    # STEP 3: REPLAY
    # ---------------------------------------------------------
    print("--- 3. Executing Graph Replay ---")
    
    new_message = engine.example(amin=-1, amax=1)
    pt_new = engine.encode(m=new_message, level=depleted_level)
    ct_new = engine.encrypt(pt=pt_new, pk=public_key, level=depleted_level)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()

    # Copy new data
    copy_data_struct(ct_new, static_input_ct)

    # Replay all graphs
    for g in graphs:
        g.replay()

    end_event.record()
    torch.cuda.synchronize()
    
    elapsed = start_event.elapsed_time(end_event)
    print(f"--- Graph Execution Time: {elapsed:.2f} ms ---")

    # ---------------------------------------------------------
    # STEP 4: VERIFY
    # ---------------------------------------------------------
    pt_dec = engine.decrypt(ct=static_output_ct, sk=secret_key)
    decoded_message = engine.decode(m=pt_dec, level=static_output_ct.level)

    error = np.abs(new_message - decoded_message).max()
    print(f"   Max Error: {error}")
    if error < 0.1:
        print("SUCCESS: Graph replay correct.")
    else:
        print("FAILURE: Graph replay incorrect.")

if __name__ == "__main__":
    test_ctos()