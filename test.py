import time
import torch
from lltm_c import LLTM as LLTM_C
from lltm_py import LLTM as LLTM_PY
from lltm_cuda import LLTM as LLTM_CUDA


assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 16
input_features = 32
state_size = 128

RNN_NAME = {
    "rnn1": "(PY)",
    "rnn2": "(C)",
    "rnn3": "(CUDA)",
}


def time_test(rnn="rnn1", device="cpu"):
    devive = device.lower()
    if device=="gpu":
        X = torch.randn(batch_size, input_features, device=cuda_device)
        h = torch.randn(batch_size, state_size, device=cuda_device)
        C = torch.randn(batch_size, state_size, device=cuda_device)

        rnn_dict = {
            "rnn1": LLTM_PY(input_features, state_size).to(cuda_device),
            "rnn2": LLTM_C(input_features, state_size).to(cuda_device),
            "rnn3": LLTM_CUDA(input_features, state_size).to(cuda_device)
        }
    else:
        X = torch.randn(batch_size, input_features)
        h = torch.randn(batch_size, state_size)
        C = torch.randn(batch_size, state_size)
        
        rnn_dict = {
            "rnn1": LLTM_PY(input_features, state_size),
            "rnn2": LLTM_C(input_features, state_size)
        }
        assert rnn != "rnn3", "rnn in ['rnn1', 'rnn2'] for host."
        
    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn_dict[rnn](X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start

    print('{}   Forward: {:.3f} s | Backward {:.3f} s  {}'.format(rnn, forward, backward, RNN_NAME[rnn]))
    

if __name__ == "__main__":
    time_test("rnn1", "cpu")
    time_test("rnn2", "cpu")
    time_test("rnn1", "gpu")
    time_test("rnn2", "gpu")
    time_test("rnn3", "gpu")
    pass
