# torch_extension
一个pytorch的预编译/即时编译案例


官网案例：https://pytorch.org/tutorials/advanced/cpp_extension.html
本项目是对torch官网案例的具体实现, 详情参考博客：https://www.cnblogs.com/dan-baishucaizi/p/16420570.html

项目背景介绍：当我们要实现一个新算子时, 可能的操作方式有：
* 基于pytorch的nn.Module类进行封装已有的算子;
* 使用C++重构;
* 使用C++和CUDA进行混合编写

这三种方式怎么实现？我们以LLTM(一个没有遗忘门的RNN结构)为例进行实验, 本项目`csrc`文件夹下实现了c++重构和混合编码两种方式, 使用python编写见`lltm_py.py`文件. 项目的整个实现流程参考上面的博客！

**项目安装**
```shell
$ python setup.py install
```

使用上面的命令安装后, 我们在共享库中就安装了lltm库, 使用这个库之前必须要先导入`torch`.

**导入**
```python
import torch
from lltm import lltm_cpp
from lltm import lltm_cuda
```

**测试**
```shell
$ python test.py
cpu  rnn1   Forward: 23.932 s | Backward 32.698 s  (PY)
cpu  rnn2   Forward: 19.476 s | Backward 57.003 s  (C)
gpu  rnn1   Forward: 16.801 s | Backward 26.295 s  (PY)
gpu  rnn2   Forward: 15.366 s | Backward 47.956 s  (C)
gpu  rnn3   Forward: 12.803 s | Backward 23.215 s  (CUDA)
```
这里C的执行结果明显异常, 若按照这个结果意味着C++比python慢, 这和我们的一般认知是不符的！可能导致的原因：环境配置(docker容器)

**官方的测试结果**
```shell
Forward: 506.480 us | Backward 444.694 us
Forward: 349.335 us | Backward 443.523 us
Forward: 187.719 us | Backward 410.815 us
Forward: 149.802 us | Backward 393.458 us
Forward: 129.431 us | Backward 304.641 us
```

这里的实验顺序和本项目的实现顺序一致！
