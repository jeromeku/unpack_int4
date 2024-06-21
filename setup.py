import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

library_name = "unpack_int4"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"


def get_extensions():
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
        ],
        "nvcc": [
            "-O3",
        ],
    }

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, f"{library_name}/csrc")
    cpp_sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    CUDA_DIR = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(CUDA_DIR, "*.cu")))
    print(f"cpp_sources: {cpp_sources}")
    print(f"cuda_sources: {cuda_sources}")

    ext_modules = [
        CUDAExtension(
            f"{library_name}._C",
            cpp_sources + cuda_sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


get_extensions()
setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Unpacking utils for torch tinygemm",
    cmdclass={"build_ext": BuildExtension},
)
