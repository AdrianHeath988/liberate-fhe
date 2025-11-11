import numpy as np
from collections import namedtuple


from liberate.liberate_fhe_cuda import mod_raise_gpu
from liberate.fhe.ckks_engine import ckks_engine
from liberate.fhe.context.ckks_context import ckks_context
from liberate.fhe import presets

Modulus64 = namedtuple('Modulus64', ['p', 'p_twice', 'p_word_size', 'p_mod_inv'])

def test_ctos():
    print("--- Setting up CKKS Context for Bootstrapping ---")

    ctx_params = {
        "buffer_bit_length":62,
        "scale_bits":40,
        "logN":15,
        "num_scales":None,
        "num_special_primes":2,
        "sigma":3.2,
        "uniform_tenary_secret":True,
        # "cache_folder":"cache/",
        "security_bits":128,
        "quantum":"post_quantum",
        'distribution':"uniform",
        "read_cache":True,
        "save_cache":True,
        "verbose":False, 
    }

    
    print()
    params = presets.params["silver"]
    print(params)

    engine = ckks_engine(verbose=True, **params)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(sk=secret_key)

    test_message = engine.example()

    level = 0

    pt = engine.encode(m=test_message, level=level)
    ct = engine.encrypt(pt=pt, pk=public_key, level=level)

    pt_dec = engine.decrypt(ct=ct, sk=secret_key)
    test_message_dec = engine.decode(m=pt_dec, level=level)

if __name__ == "__main__":
    # You'll need to define the Modulus64 struct in a way Python can access it.
    # One way is to use ctypes or create a simple class.
    from collections import namedtuple
    Modulus64 = namedtuple('Modulus64', ['p', 'p_twice', 'p_word_size', 'p_mod_inv'])

    test_ctos()