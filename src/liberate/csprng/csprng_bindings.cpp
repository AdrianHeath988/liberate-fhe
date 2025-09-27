#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for each specific binding function
// Each of these will be defined in its corresponding .cpp file (e.g., chacha20.cpp)
void bind_chacha20(py::module_ &m);
void bind_discrete_gaussian(py::module_ &m);
void bind_randint(py::module_ &m);
void bind_randround(py::module_ &m);

// Main binding function for the entire csprng submodule
void bind_csprng_modules(py::module_ &m) {
    // Create a submodule for csprng to keep things organized
    auto csprng_submodule = m.def_submodule("csprng", "CSPRNG functions");

    // Call the individual binding functions to populate the submodule
    bind_chacha20(csprng_submodule);
    bind_discrete_gaussian(csprng_submodule);
    bind_randint(csprng_submodule);
    bind_randround(csprng_submodule);
}