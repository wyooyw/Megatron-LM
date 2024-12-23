import torch
import gc
from functools import partial


# a = [1,2,3]
# print(a.__getitem__(0))
# exit()

class Tensor:
    def __init__(self, tensor):
        self.tensor = [tensor]

def show_memory():
    memory = torch.cuda.memory_allocated()
    memory_mb = memory / 1024 / 1024
    print(f"memory: {memory_mb:2f} MB")
    # torch.cuda.reset_peak_memory_stats()
    return memory_mb

# show_memory()
n = 128

gc.set_threshold(1, 1, 1)
gc.disable()
print(f"{gc.get_threshold()=}")
# torch.cuda.cudart().cudaProfilerStart()
a1 = [torch.randn([2, 16384,16384], dtype=torch.float32, device="cuda")]
a2 = [torch.randn([2, 16384,16384], dtype=torch.float32, device="cuda")]
a3 = [torch.randn([2, 16384,16384], dtype=torch.float32, device="cuda")]
a4 = [torch.randn([2, 16384,16384], dtype=torch.float32, device="cuda")]

show_memory()

def test_bmm(a1, a2, a3, a4):
    b1 = a1[0]; del a1[0]
    b2 = a2[0]; del a2[0]
    b3 = a3[0]; del a3[0]
    b4 = a4[0]; del a4[0]
    del b1
    del b2
    del b3
    del b4
    # a12 = torch.bmm(a1, a2); del a1, a2
    # a34 = torch.bmm(a3, a4); del a3, a4
fn = partial(test_bmm, a1, a2, a3, a4)
del a1
del a2
del a3
del a4
show_memory()
fn()
show_memory()
del fn
show_memory()
# torch.cuda.cudart().cudaProfilerStop()