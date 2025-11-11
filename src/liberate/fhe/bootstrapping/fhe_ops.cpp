#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cuda_runtime.h>
#include "liberate/ntt/ntt.h"
#include "mod_raise_kernel.h"

#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        throw std::runtime_error(\
            std::string("CUDA error: ") + cudaGetErrorString(err_) + \
            " in file " + __FILE__ + " at line " + std::to_string(__LINE__)); \
    } \
}



namespace py = pybind11;
// This wrapper orchestrates the GPU operations
py::tuple mod_raise_gpu(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ct_in,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ntt_table,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> intt_table,
    py::array_t<Modulus64, py::array::c_style | py::array::forcecast> moduli,
    int n_power,
    int q_size,
    int p_size
) {
    // 1. Get pointers and context info
    auto ct_in_buf = ct_in.request();
    auto ntt_table_buf = ntt_table.request();
    auto intt_table_buf = intt_table.request();
    auto moduli_buf = moduli.request();

    const uint64_t* h_ct_in = static_cast<const uint64_t*>(ct_in_buf.ptr);
    const uint64_t* h_ntt_table = static_cast<const uint64_t*>(ntt_table_buf.ptr);
    const uint64_t* h_intt_table = static_cast<const uint64_t*>(intt_table_buf.ptr);
    const Modulus64* h_moduli = static_cast<const Modulus64*>(moduli_buf.ptr);

    int n = 1 << n_power;
    size_t ct_in_size_bytes = ct_in_buf.size * sizeof(uint64_t);
    size_t ct_raised_size = 2 * n * (q_size + p_size);
    size_t ct_raised_size_bytes = ct_raised_size * sizeof(uint64_t);

    cudaStream_t stream = 0; // Use default stream

    // 2. Allocate GPU memory
    uint64_t *d_ct_in, *d_ct_intt, *d_ct_raised;
    uint64_t *d_ntt_table, *d_intt_table;
    Modulus64* d_moduli;

    CUDA_CHECK(cudaMalloc(&d_ct_in, ct_in_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_ct_intt, ct_in_size_bytes)); // Same size as input
    CUDA_CHECK(cudaMalloc(&d_ct_raised, ct_raised_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_ntt_table, ntt_table.nbytes()));
    CUDA_CHECK(cudaMalloc(&d_intt_table, intt_table.nbytes()));
    CUDA_CHECK(cudaMalloc(&d_moduli, moduli.nbytes()));

    // 3. Copy input data from Host (CPU) to Device (GPU)
    CUDA_CHECK(cudaMemcpyAsync(d_ct_in, h_ct_in, ct_in_size_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ntt_table, h_ntt_table, ntt_table.nbytes(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_intt_table, h_intt_table, intt_table.nbytes(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_moduli, h_moduli, moduli.nbytes(), cudaMemcpyHostToDevice, stream));

    // 4. Call the INVERSE NTT function (forward=false)
    ntt_kernel_wrapper(d_ct_in, d_ct_intt, d_intt_table, (uint64_t*)d_moduli, n_power, false, stream);

    // 5. Launch the new mod_raise_kernel
    // The wrapper function computes the grid/block and launches the kernel.
    mod_raise_kernel_wrapper(d_ct_intt, d_ct_raised, d_moduli, n_power, q_size, p_size, stream);

    // 6. Call the FORWARD NTT function (forward=true) on the raised ciphertext
    // This is done in-place.
    ntt_kernel_wrapper(d_ct_raised, d_ct_raised, d_ntt_table, (uint64_t*)d_moduli, n_power, true, stream);
    
    // 7. Copy the final result from Device (GPU) back to Host (CPU)
    auto result_array = py::array_t<uint64_t>(ct_raised_size);
    auto result_buf = result_array.request();
    uint64_t* h_result = static_cast<uint64_t*>(result_buf.ptr);

    CUDA_CHECK(cudaMemcpyAsync(h_result, d_ct_raised, ct_raised_size_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 8. Free GPU memory
    CUDA_CHECK(cudaFree(d_ct_in));
    CUDA_CHECK(cudaFree(d_ct_intt));
    CUDA_CHECK(cudaFree(d_ct_raised));
    CUDA_CHECK(cudaFree(d_ntt_table));
    CUDA_CHECK(cudaFree(d_intt_table));
    CUDA_CHECK(cudaFree(d_moduli));

    // The result array is flat; reshape it in Python to (2, num_primes, N)
    return py::make_tuple(result_array);
}

void bind_bootstrapping_modules(py::module_ &m) {
    m.def("mod_raise_gpu", &mod_raise_gpu,
          "Performs the modulus raising operation (INTT -> mod_raise -> NTT) on the GPU",
          py::arg("ct_in"), py::arg("ntt_table"), py::arg("intt_table"),
          py::arg("moduli"), py::arg("n_power"), py::arg("q_size"), py::arg("p_size"));
}


