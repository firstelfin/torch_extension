from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['asrc/lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
