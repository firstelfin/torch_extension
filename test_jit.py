from torch.utils.cpp_extension import load

__call__ = ["lltm_cpp"]

lltm_cpp = load(name="lltm_cpp", sources=["csrc/lltm.cpp"])
