#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cuda_runtime.h>
#include "liberate/ntt/ntt.h"
#include "mod_raise_kernel.h"
#include "ctos.h"
#include <iostream>

#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        throw std::runtime_error(\
            std::string("CUDA error: ") + cudaGetErrorString(err_) + \
            " in file " + __FILE__ + " at line " + std::to_string(__LINE__)); \
    } \
}



namespace py = pybind11;
py::tuple mod_raise_gpu(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ct_in,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ntt_table,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> intt_table,
    py::array_t<Modulus64, py::array::c_style | py::array::forcecast> moduli,
    int n_power,
    int q_size,
    int p_size
) {
    
}


py::array_t<uint64_t> ctos_gpu(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ct_in,
    py::array_t<Modulus64, py::array::c_style | py::array::forcecast> moduli,
    int n_power,
    int q_size,
    int p_size
) {
    // Request buffer info
    auto in_buf = ct_in.request();
    size_t total_elems = static_cast<size_t>(in_buf.size);

    // Prepare output with same shape as input
    std::vector<ssize_t> shape = in_buf.shape;
    py::array_t<uint64_t> out(shape);
    auto out_buf = out.request();

    // Handle empty input
    if (total_elems == 0) {
        return out;
    }

    auto mod_buf = moduli.request();
    size_t mod_count = static_cast<size_t>(mod_buf.size);
    if (mod_count == 0) {
        throw std::runtime_error("moduli array must not be empty");
    }

    // Device pointers
    uint64_t *d_in = nullptr, *d_out = nullptr;
    Modulus64 *d_mod = nullptr;

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, total_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, total_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_mod, mod_count * sizeof(Modulus64)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpyAsync(d_in, in_buf.ptr, total_elems * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mod, mod_buf.ptr, mod_count * sizeof(Modulus64),
                               cudaMemcpyHostToDevice, stream));

    // Call kernel wrapper (must be implemented in ctos.cu)
    ctos_kernel_wrapper(d_in, d_out, d_mod, n_power, q_size, p_size, total_elems, stream);

    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(out_buf.ptr, d_out, total_elems * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));

    // Synchronize and cleanup
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_mod));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return out;
}






void bind_bootstrapping_modules(py::module_ &m) {
    std::cout << "In bind_bootstrapping_modules" << std::endl;

    // Create a 'bootstrapping' submodule
    auto boot_submodule = m.def_submodule("bootstrapping", "Bootstrapping functions");

    // Register functions on the bootstrapping submodule (minimal binding form)
    boot_submodule.def("mod_raise_gpu", &mod_raise_gpu, "Performs the modulus raising operation (INTT -> mod_raise -> NTT) on the GPU");
    boot_submodule.def("ctos_gpu", &ctos_gpu, "Performs the CTOS operation on the GPU");
}


