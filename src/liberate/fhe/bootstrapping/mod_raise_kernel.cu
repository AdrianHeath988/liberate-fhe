#include <stdint.h>

struct Modulus64 {
    uint64_t p;
    uint64_t p_twice;
    uint64_t p_word_size;
    uint64_t p_mod_inv;
};

// Simplified modular reduction function
__device__ uint64_t reduce_forced(uint64_t input, Modulus64 mod) {
    return input % mod.p;
}

// The new kernel
__global__ void mod_raise_kernel(const uint64_t* input, uint64_t* output,
                                 const Modulus64* modulus, int n_power, int q_size, int p_size)
{
    int n = 1 << n_power;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Coefficient index
    int idy = blockIdx.y; // Target RNS prime index (for p_primes)
    int idz = blockIdx.z; // Ciphertext part index (c0 or c1)

    if (idx >= n) return;

    // Location of the input coefficient (after INTT)
    // Assumes input is from the lowest level, so only one RNS component.
    int location_input = idx + (idz * n);

    // Location for the output coefficient in the new extended basis
    int location_output = idx + (idy * n) + (idz * (p_size * n));

    uint64_t input_val = input[location_input];
    // Reduce the coefficient by the new prime
    output[location_output] = reduce_forced(input_val, modulus[q_size + idy]);
}