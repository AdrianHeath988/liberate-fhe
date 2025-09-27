#include <stdint.h>
#include "mod_raise_kernel.h"

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
void mod_raise_kernel_wrapper(const unsigned long* ct_in,
                              unsigned long* ct_out,
                              const Modulus64* moduli,
                              int num_polys,
                              int num_moduli,
                              int N,
                              cudaStream_t stream) {
    const dim3 threads(N);
    const dim3 blocks(num_polys);

    mod_raise_kernel<<<blocks, threads, 0, stream>>>(ct_in, ct_out, moduli, num_polys, num_moduli, N);
}
