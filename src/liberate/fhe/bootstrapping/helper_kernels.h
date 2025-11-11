#ifndef HELPER_H
#define HELPER_H

#include <stdint.h>
#include <cuda_runtime.h>

// Define a structure for the modulus, which can be passed to the kernel
struct Modulus64 {
    uint64_t p;
    uint64_t p_twice;
    uint64_t p_word_size;
    uint64_t p_mod_inv;
};

// Declaration of the CUDA kernel
void mod_raise_kernel_wrapper(
    const uint64_t* input,
    uint64_t* output,
    const Modulus64* modulus,
    int n_power,
    int q_size,
    int p_size,
    cudaStream_t stream
);

#endif 