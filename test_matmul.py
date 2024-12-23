import pycuda.autoinit
import pycuda.driver as cuda
import torch
import time

def benchmark_pytorch_fp16(M,N,K, num_runs):
    # 确保使用 GPU 并设置数据类型为半精度浮点数 (float16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    # 生成随机矩阵
    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)    
    # 预热 GPU，进行一次矩阵乘法
    C = torch.matmul(A, B)    
    # 记录开始时间
    start_time = time.time()    
    # 多次进行矩阵乘法，计算 FLOPS
    start = cuda.Event()
    end = cuda.Event()
    start.record()    
    for _ in range(num_runs):
        C = torch.mm(A, B)    
    end.record()
    torch.cuda.synchronize()    
    elapsed_time = start.time_till(end) / num_runs    
    # 计算 GFLOPS
    num_operations = 2 * M*N*K
    gflops = num_operations / (elapsed_time * 1e-3) / 1e12    
    return elapsed_time, gflops
    # 记录结束时间
    end_time = time.time()    
    # 计算平均运行时间
    elapsed_time = (end_time - start_time) / num_runs    
    # 计算总的 FLOPs
    total_flops = 2 * M*K*N    
    # 计算 GFLOPS
    gflops = total_flops / elapsed_time / 1e12    
    return elapsed_time, gflops
# 设置矩阵大小和运行次数
num_runs = 32
M=2048
N=2048
K=40960
for i in range(5):
    # 运行基准测试
    elapsed_time, gflops = benchmark_pytorch_fp16(M,N,K, num_runs)
    # 输出结果
    print(f"Num:{i} 矩阵乘法大小: {M}x{K}X{N} 平均运行时间: {elapsed_time:.6f} 秒 TFLOPS: {gflops:.2f}")
    time.sleep(0.1)
# EOFxflops.py
