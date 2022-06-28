import time
import torch
from lltm_c import LLTM as LLTM_C
from lltm_py import LLTM as LLTM_PY

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features, device=cuda_device)
h = torch.randn(batch_size, state_size, device=cuda_device)
C = torch.randn(batch_size, state_size, device=cuda_device)

rnn_dict = {
    "rnn1": LLTM_PY(input_features, state_size).to(cuda_device),
    "rnn2": LLTM_C(input_features, state_size).to(cuda_device)
}
rnn_name = {
    "rnn1": "(PY)",
    "rnn2": "(C)"
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
        torch.cuda.synchronize()
        backward += time.time() - start

    print('{} Forward: {:.3f} s | Backward {:.3f} s  {}'.format(rnn, forward, backward, rnn_name[rnn]))
    

if __name__ == "__main__":
  time_test()
  time_test("rnn2")
  pass
