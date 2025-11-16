// Minimal CTOS kernel: copy input to output
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "ctos.h"

// Simple device kernel that copies uint64_t elements
__global__ void ctos_copy_kernel(const uint64_t* in, uint64_t* out, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		out[idx] = in[idx];
	}
}

// Host-visible wrapper called from the C++ code. Launches the copy kernel on the provided stream.
void ctos_kernel_wrapper(
	const uint64_t* input,
	uint64_t* output,
	const Modulus64* /* moduli */,
	int /* n_power */,
	int /* q_size */,
	int /* p_size */,
	size_t length,
	cudaStream_t stream
) {
	if (length == 0) return;

	const int block = 256;
	int grid = static_cast<int>((length + block - 1) / block);

	// Launch kernel on provided stream
	ctos_copy_kernel<<<grid, block, 0, stream>>>(input, output, length);

	// Do not synchronize here; caller synchronizes the stream after the wrapper returns.
}

