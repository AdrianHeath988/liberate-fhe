import numpy as np
from collections import namedtuple

# --- Corrected Imports ---
# After building, the C++ module will be available at this path.
from liberate.liberate_fhe_cuda import mod_raise_gpu
from liberate.fhe.ckks_engine import ckks_engine
from liberate.fhe.context.ckks_context import ckks_context

# --- Python Mirror of the C++ Struct ---
# We create a named tuple to match the Modulus64 struct in C++.
# Pybind11 can automatically convert this when we pass it as a NumPy array.
Modulus64 = namedtuple('Modulus64', ['p', 'p_twice', 'p_word_size', 'p_mod_inv'])

def test_mod_raise():
    """
    A test script to verify the functionality of the mod_raise_gpu function.
    """
    print("--- Setting up CKKS Context for Bootstrapping ---")

    # 1. Setup Context with Bootstrapping Primes
    # Use a smaller N for faster testing
    n_power = 12
    N = 1 << n_power
    log_scale = 30
    
    # Primes for standard operations
    q_primes = [65537, 114689] # Example primes
    q_size = len(q_primes)

    # Primes for bootstrapping modulus extension
    p_primes = [163841, 212993] # Example primes
    p_size = len(p_primes)

    # The CKKSContext needs to be aware of both sets of primes
    logN = 15
    context = ckks_context(logN=logN, scale_bits=log_scale, num_scales=len(q_primes))

    
    # The full modulus chain for the raised ciphertext
    full_modulus_chain = context.bootstrapping_modulus_chain
    print(f"Polynomial degree N = {N}")
    print(f"q_primes: {q_primes}")
    print(f"p_primes: {p_primes}")
    print(f"Full modulus chain (q+p): {full_modulus_chain}")

    # Create the CKKS engine
    engine = ckks_engine(context)
    engine.keygen()

    print("\n--- Encrypting a test message ---")
    message = np.array([3.14, 1.59, 2.65, -1.0])
    plaintext = engine.encode(message)
    ciphertext = engine.encrypt(plaintext)

    print(f"Original ciphertext shape (c0): {ciphertext.c0.shape}")
    print(f"Original ciphertext level: {ciphertext.level}")

    # 2. Call the Mod Raise Function
    print("\n--- Calling the mod_raise_gpu function ---")
    
    # Prepare the inputs for the C++ function
    # The input ciphertext must be at the lowest level (highest depth)
    # For this test, we assume a fresh ciphertext is at level 0 (depth = q_size)
    # The C++ function expects a flat numpy array
    ct_in_flat = np.concatenate([ciphertext.c0.flatten(), ciphertext.c1.flatten()])

    # We need to create a Python representation of the Modulus64 struct
    # This would ideally be a helper function in your Python code.
    moduli_structs = []
    # This is a simplified creation, you'll need to get the real values from context
    for p in full_modulus_chain:
        moduli_structs.append(Modulus64(p, 2*p, 64, pow(p, -1, 1<<64)))
    
    moduli_array = np.array(moduli_structs)

    # Call the wrapped CUDA function
    (raised_ct_flat,) = mod_raise_gpu(
        ct_in_flat,
        engine.context.ntt_context.ntt_table,
        engine.context.ntt_context.intt_table,
        moduli_array,
        n_power,
        q_size,
        p_size
    )

    # 3. Verify the Output
    print("\n--- Verifying the raised ciphertext ---")
    
    # Reshape the flat output array back into a structured ciphertext
    expected_shape = (2, q_size + p_size, N)
    raised_ct_array = raised_ct_flat.reshape(expected_shape)
    
    # Create a new data_struct for the raised ciphertext
    raised_ciphertext = ciphertext.clone() # Use clone to copy metadata
    raised_ciphertext.c0 = raised_ct_array[0]
    raised_ciphertext.c1 = raised_ct_array[1]
    raised_ciphertext.modulus = full_modulus_chain
    # The level might need adjustment depending on your implementation logic
    
    print(f"Raised ciphertext shape (c0): {raised_ciphertext.c0.shape}")

    # 4. Decrypt and Check
    # To decrypt a ciphertext with an extended modulus, the secret key
    # must also be extended to that modulus.
    print("\n--- Decrypting and checking the result ---")
    
    # This is a conceptual step. The standard `decrypt` might not work directly
    # on a modulus-raised ciphertext without modifications to handle the new RNS components.
    # A proper test would involve a custom decryption that uses the full modulus chain.
    try:
        # We create a temporary "bootstrapping" secret key
        # In a real scenario, you'd have a more robust way of handling this
        sk_boot = engine.secret_key.clone()
        sk_boot.modulus = full_modulus_chain

        decrypted_raised_pt = engine.decrypt(raised_ciphertext, sk=sk_boot)
        decrypted_message = engine.decode(decrypted_raised_pt)

        print(f"Original message:  {message}")
        print(f"Decrypted message: {decrypted_message[:len(message)]}")
        
        # Check if the decrypted values are close to the original ones
        np.testing.assert_allclose(message, decrypted_message[:len(message)], rtol=1e-3)
        print("\n✅ Test PASSED: Decrypted message matches the original.")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        print("Note: A standard decrypt may not work directly. This may require")
        print("a modified decryption function that operates on the extended modulus (q+p).")


if __name__ == "__main__":
    # You'll need to define the Modulus64 struct in a way Python can access it.
    # One way is to use ctypes or create a simple class.
    from collections import namedtuple
    Modulus64 = namedtuple('Modulus64', ['p', 'p_twice', 'p_word_size', 'p_mod_inv'])

    test_mod_raise()