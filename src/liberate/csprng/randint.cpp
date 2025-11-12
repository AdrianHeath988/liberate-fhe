#include <torch/extension.h>
#include <vector>

// Check types.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x)


// Forward declaration.
void randint_cuda(torch::Tensor rand_bytes, uint64_t *q);
torch::Tensor randint_fast_cuda(torch::Tensor states, uint64_t *q, int64_t shift, size_t step);



// The main function.
//----------------
// Normal version.
void randint(std::vector<torch::Tensor> inputs,
             std::vector<size_t> q_ptrs) {

    for (auto i=0; i<inputs.size(); i++){
        CHECK_INPUT(inputs[i]);

        // reinterpret pointers from numpy.
        uint64_t *q = reinterpret_cast<uint64_t*>(q_ptrs[i]);
        
        // Run in cuda.
        randint_cuda(inputs[i], q);
    }
}

//--------------
// Fast version.
std::vector<torch::Tensor> randint_fast(
    std::vector<torch::Tensor> states,
    std::vector<size_t> q_ptrs,
    int64_t shift,
    size_t step) {

    std::vector<torch::Tensor> outputs;
    
    for (auto i=0; i<states.size(); i++){
        uint64_t *q = reinterpret_cast<uint64_t*>(q_ptrs[i]);
        auto result = randint_fast_cuda(states[i], q, shift, step);
        outputs.push_back(result);
    }
    return outputs;
}

void bind_randint(py::module_ &m) {
    m.def("randint", &randint, "Generate random integers uniformly in [0, modulus)");
    m.def("randint_fast", &randint_fast, "Generate random integers (fast version)");
}