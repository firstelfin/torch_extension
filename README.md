# torch_extension
一个pytorch的预编译/即时编译案例

本项目是对torch官网案例的具体实现, 详情参考博客：https://www.cnblogs.com/dan-baishucaizi/p/16420570.html

**项目安装**
```shell
$ python setup.py install
```

使用上面的命令安装后, 我们在共享库中就安装了lltm库, 使用这个库之前必须要先导入`torch`.

导入
```python
import torch
from lltm import lltm_cpp
from lltm import lltm_cuda
```

测试
```shell
$ python test.py

```
