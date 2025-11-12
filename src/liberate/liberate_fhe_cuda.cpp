#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of the functions that define bindings
// for each submodule (csprng, ntt, bootstrapping).
void bind_csprng_modules(py::module_ &m);
void bind_ntt_modules(py::module_ &m);
void bind_bootstrapping_modules(py::module_ &m);

// The one and only PYBIND11_MODULE definition
PYBIND11_MODULE(liberate_fhe_cuda, m) {
    m.doc() = "Unified CUDA backend for Liberate.FHE";

    // Call the binding functions to add all submodules
    bind_csprng_modules(m);
    bind_ntt_modules(m);
    bind_bootstrapping_modules(m);
}