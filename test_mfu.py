import torch
from functools import partial
import numpy as np

def profile_fn(n_warmup, n_test, fn):
    # warmup
    with torch.no_grad():  # 确保不会跟踪这些操作的梯度
        for i in range(n_warmup):
            fn()

    # 创建CUDA事件列表来记录每一轮的开始和结束时间
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_test)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_test)]

    # 记录每一轮操作的开始和结束时间
    with torch.no_grad():  # 确保不会跟踪这些操作的梯度
        for i in range(n_test):
            start_events[i].record()
            fn()
            end_events[i].record()

    # 同步所有事件
    for event in end_events:
        event.synchronize()

    # 计算每一轮操作的耗时（单位：毫秒）
    elapsed_times_ms = []
    for i in range(n_test):
        elapsed_time_ms = start_events[i].elapsed_time(end_events[i])
        elapsed_times_ms.append(elapsed_time_ms)
        # print(f"Iteration {i+1} elapsed time: {elapsed_time_ms} ms")
    
    elapsed_times_ms = np.array(elapsed_times_ms)
    return elapsed_times_ms.mean().item(), elapsed_times_ms.std().item()
    

def test_matmul(n, k, m, dtype, n_warmup = 10, n_test = 10):
    lhs = torch.randn(n, k, dtype=dtype, device="cuda")
    rhs = torch.randn(k, m, dtype=dtype, device="cuda")
    matmul_fn = partial(torch.matmul, lhs, rhs)
    mean_time, std = profile_fn(n_warmup, n_test, matmul_fn)
    flops = n * k * m * 2
    througput = flops / mean_time * 1000 / 10**12
    peak_throughput = {
        torch.bfloat16: 1513,
        torch.float32: 756
    }[dtype]

    mfu = througput / peak_throughput * 100

    print(f"{n=}, {k=}, {m=}, {dtype=}")
    print(f"    {mean_time=}ms")
    print(f"    {flops=}")
    print(f"    {througput=:3} TFLOPS")
    print(f"    {mfu=:3}%")
    print("")

    # return mean_time, flops, througput, mfu

if __name__=="__main__":
    # for i in [8192,16384,32768]:
    test_matmul(2048, 65536 * 4, 2048, torch.bfloat16)