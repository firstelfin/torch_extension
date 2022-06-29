from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='lltm',
    ext_modules=[CppExtension('lltm_cpp', ['csrc/lltm.cpp']), CUDAExtension("lltm_cuda", ["csrc/lltm_cuda.cpp", "csrc/lltm_cuda_kernel.cu"])],
    cmdclass={'build_ext': BuildExtension}
)
