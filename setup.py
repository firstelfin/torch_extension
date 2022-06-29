from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='lltm',
    version="0.0.1",
    author='firstelfin',
    author_email='firstelfin@qq.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/firstelfin/torch_extension",
    ext_modules=[CppExtension('lltm.lltm_cpp', ['csrc/lltm.cpp']), CUDAExtension("lltm.lltm_cuda", ["csrc/lltm_cuda.cpp", "csrc/lltm_cuda_kernel.cu"])],
    cmdclass={'build_ext': BuildExtension},
    py_modules=["test_cpu", "test_gpu", "test_jit", "lltm_c", "lltm_cuda", "lltm_py"],
    install_requires=["torch"],
    keywords="extensions, cpp_extensions"
)
