#ifndef CTOS_H
#define CTOS_H

#include <stdint.h>
#include <cuda_runtime.h>
#include "helper_kernels.h"
// Declaration of the CTOS kernel wrapper to be implemented in ctos.cu
#include <cstddef>

// length is the number of uint64_t elements to process in input/output
void ctos_kernel_wrapper(
	const uint64_t* input,
	uint64_t* output,
	const Modulus64* moduli,
	int n_power,
	int q_size,
	int p_size,
	size_t length,
	cudaStream_t stream
);
#endif