import time
import torch
from lltm_c import LLTM as LLTM_C
from lltm_py import LLTM as LLTM_PY

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn_dict = {
    "rnn1": LLTM_PY(input_features, state_size),
    "rnn2": LLTM_C(input_features, state_size)
}


def time_test(rnn="rnn1"):
    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn_dict[rnn](X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start

    print('{} Forward: {:.3f} s | Backward {:.3f} s'.format(rnn, forward, backward))
    

if __name__ == "__main__":
  time_test()
  time_test("rnn2")
  pass
