#ifndef NTT_H
#define NTT_H

#include <pybind11/pybind11.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace py = pybind11;

// This declares the NTT wrapper function, making it visible to other files.
// updated ntt.h
void ntt_kernel_wrapper(const unsigned long* a,
                        unsigned long* b,
                        const unsigned long* psi,
                        const unsigned long* moduli, // Add this parameter
                        int N,
                        bool forward,
                        cudaStream_t stream);

// This declares the Python binding function for the NTT module.
void bind_ntt_modules(py::module_ &m);

#endif // NTT_H