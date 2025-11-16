from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = [
    CUDAExtension(
        # The full import path of the C++ module
        name="liberate.liberate_fhe_cuda",
        sources=[
            # Main entry point that defines the Python module
            "src/liberate/liberate_fhe_cuda.cpp",
            # Binder for the csprng functions
            "src/liberate/csprng/csprng_bindings.cpp",
            # All other C++/CUDA source files
            "src/liberate/csprng/randint.cpp",
            "src/liberate/csprng/randint_cuda_kernel.cu",
            "src/liberate/csprng/randround.cpp",
            "src/liberate/csprng/randround_cuda_kernel.cu",
            "src/liberate/csprng/discrete_gaussian.cpp",
            "src/liberate/csprng/discrete_gaussian_cuda_kernel.cu",
            "src/liberate/csprng/chacha20.cpp",
            "src/liberate/csprng/chacha20_cuda_kernel.cu",
            "src/liberate/ntt/ntt.cpp",
            "src/liberate/ntt/ntt_cuda_kernel.cu",
            "src/liberate/fhe/bootstrapping/fhe_ops.cpp",
            "src/liberate/fhe/bootstrapping/mod_raise_kernel.cu",
            "src/liberate/fhe/bootstrapping/ctos.cu",
        ],
        include_dirs=["src/"]
    )
]





if __name__ == "__main__":
    setup(
        name="liberate-fhe",
        version="0.1.0",
        # find_packages() automatically discovers your Python packages (e.g., 'liberate')
        packages=find_packages(where="src"),
        # This tells setuptools that the package lives in the 'src' directory
        package_dir={"": "src"},
        # This lists the C++ extensions to be compiled
        ext_modules=ext_modules,
        # This command class from PyTorch handles the CUDA compilation
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
