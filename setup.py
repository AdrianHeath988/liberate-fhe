from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="randint_cuda",
        sources=[
            "src/liberate/csprng/randint.cpp",
            "src/liberate/csprng/randint_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            "src/liberate/csprng/randround.cpp",
            "src/liberate/csprng/randround_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            "src/liberate/csprng/discrete_gaussian.cpp",
            "src/liberate/csprng/discrete_gaussian_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            "src/liberate/csprng/chacha20.cpp",
            "src/liberate/csprng/chacha20_cuda_kernel.cu",
        ],
    ),
]

ext_modules_ntt = [
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            "src/liberate/ntt/ntt.cpp",
            "src/liberate/ntt/ntt_cuda_kernel.cu",
        ],
    )
]

# --- New Module for Bootstrapping ---
ext_modules_bootstrapping = [
    CUDAExtension(
        name="fhe_ops_cuda",
        sources=[
            "src/liberate/fhe/bootstrapping/fhe_ops.cpp",
            "src/liberate/fhe/bootstrapping/mod_raise_kernel.cu",
            # Include NTT sources to resolve link-time dependencies
            "src/liberate/ntt/ntt.cpp",
            "src/liberate/ntt/ntt_cuda_kernel.cu",
        ],
        # Add include directory so C++ can find the header files
        include_dirs=["src/liberate/"]
    )
]


if __name__ == "__main__":
    setup(
        name="csprng",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext"],
        options={
            "build": {
                "build_lib": "src/liberate/csprng",
            }
        },
    )

    setup(
        name="ntt",
        ext_modules=ext_modules_ntt,
        script_args=["build_ext"],
        cmdclass={"build_ext": BuildExtension},
        options={
            "build": {
                "build_lib": "src/liberate/ntt",
            }
        },
    )

    # --- New Setup Call for Bootstrapping ---
    setup(
        name="bootstrapping",
        ext_modules=ext_modules_bootstrapping,
        script_args=["build_ext"],
        cmdclass={"build_ext": BuildExtension},
        options={
            "build": {
                "build_lib": "src/liberate/fhe/bootstrapping",
            }
        },
    )