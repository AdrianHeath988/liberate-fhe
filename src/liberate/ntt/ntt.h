#ifndef NTT_H
#define NTT_H

#include <pybind11/pybind11.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace py = pybind11;

// This declares the NTT wrapper function, making it visible to other files.
void ntt_kernel_wrapper(const uint64_t* in, uint64_t* out,
                        const uint64_t* r, const uint64_t* m,
                        int n_power, bool forward, cudaStream_t stream);

// This declares the Python binding function for the NTT module.
void bind_ntt_modules(py::module_ &m);

#endif // NTT_H