import torch
import numpy as np
from functools import partial
from torch.distributed import ReduceOp
from megatron.core.extensions.wyo.graph.utils import print_rank_0

from megatron.core.extensions.wyo.model.communicate.communicate import (
    gather_along_first_dim_in_tp_group,
    reduce_scatter_along_first_dim_in_tp_group,
    all_reduce_in_tp_group,
)

class BasicProfiler:
    def __init__(self):
        self.n_warmup = 20
        self.n_test = 20
        self.throughput = dict()

    def begin_basic_profile(self):
        """
        profile to get throuput
        """
        num_bytes = 8388608 * 16
        self.throughput["gather_along_first_dim_in_tp_group"] = self._profile_communication(num_bytes, "gather_along_first_dim_in_tp_group")
        self.throughput["reduce_scatter_along_first_dim_in_tp_group"] = self._profile_communication(num_bytes, "reduce_scatter_along_first_dim_in_tp_group")
        self.throughput["all_reduce_in_tp_group"] = self._profile_communication(num_bytes, "all_reduce_in_tp_group")
        print_rank_0(f"{self.throughput['gather_along_first_dim_in_tp_group']=}")
        print_rank_0(f"{self.throughput['reduce_scatter_along_first_dim_in_tp_group']=}")
        print_rank_0(f"{self.throughput['all_reduce_in_tp_group']=}")
        # exit()
        self.throughput["compute_bf16"] = self._profile_compute(torch.bfloat16)

        print_rank_0("profiler throughput:")
        print_rank_0(f"  gather_along_first_dim_in_tp_group: {self.throughput['gather_along_first_dim_in_tp_group']} bytes / ms")
        print_rank_0(f"  reduce_scatter_along_first_dim_in_tp_group: {self.throughput['reduce_scatter_along_first_dim_in_tp_group']} bytes / ms")
        print_rank_0(f"  all_reduce_in_tp_group: {self.throughput['all_reduce_in_tp_group']} bytes / ms")
        print_rank_0(f"  compute_bf16: {self.throughput['compute_bf16']} mac / ms")


    def get_time(self, op_name, inputs):
        if op_name in [
            "gather_along_first_dim_in_tp_group", 
            "reduce_scatter_along_first_dim_in_tp_group", 
            "all_reduce_in_tp_group"
        ]:
            tensor = inputs[0]
            size = tensor.element_size() * tensor.numel()
            predict_time = size / self.throughput[op_name]
        
        elif op_name == "mm":
            tensor_a, tensor_b = inputs[0], inputs[1]
            m,k = tensor_a.shape
            k,n = tensor_b.shape
            flops = m * k * n
            if tensor_a.dtype==torch.bfloat16:
                predict_time = flops / self.throughput["compute_bf16"]
            else:
                assert False, f"Should not happen! {tensor_a.dtype=}"

        elif op_name == "flash_attn":
            query = inputs[0]
            b, s, nh, dh = query.shape
            flops_qk = (s * dh * s) * b * nh
            flops_sv = (s * dh * s) * b * nh
            flops = flops_qk + flops_sv
            predict_time = flops / self.throughput["compute_bf16"]

        elif op_name == "flash_attn_bwd":
            query = inputs[1]
            b, s, nh, dh = query.shape
            flops_qk = (s * dh * s) * b * nh
            flops_sv = (s * dh * s) * b * nh
            flops = flops_qk + flops_sv
            fwd_predict_time = flops / self.throughput["compute_bf16"]
            predict_time = fwd_predict_time * 2

        else:
            predict_time = 0

        # print_rank_0(f"get_time {op_name=}, {predict_time=}ms")
        return predict_time


    def _profile_communication(self, data_bytes, comm_name):
        input_tensor = torch.randn(data_bytes // 4, dtype=torch.float32).cuda()
        comm_fn = {
            "gather_along_first_dim_in_tp_group":gather_along_first_dim_in_tp_group,
            "reduce_scatter_along_first_dim_in_tp_group":reduce_scatter_along_first_dim_in_tp_group,
            "all_reduce_in_tp_group":all_reduce_in_tp_group,
        }[comm_name]

        comm_fn = partial(comm_fn, _inp=input_tensor, async_op=False)
        time_mean, time_std = profile_fn(self.n_warmup, self.n_test, comm_fn)
        # mean between gpus
        time_mean = torch.Tensor([time_mean]).cuda()
        torch.distributed.all_reduce(time_mean)
        time_mean = time_mean / torch.distributed.get_world_size()
        time_mean = time_mean.item()

        print_rank_0(f"{comm_name=}, {time_mean=}ms")

        throughput = data_bytes / time_mean
        return throughput

    def _profile_compute(self, dtype):
        a = torch.randn([8192, 8192], dtype=dtype, device="cuda")
        b = torch.randn([8192, 8192], dtype=dtype, device="cuda")
        flops = 8192 * 8192 * 8192
        fn = partial(torch.matmul, input=a, other=b)
        time_mean, time_std = profile_fn(self.n_warmup, self.n_test, fn)

        time_mean = torch.Tensor([time_mean]).cuda()
        torch.distributed.all_reduce(time_mean)
        time_mean = time_mean / torch.distributed.get_world_size()
        time_mean = time_mean.item()

        throughput = flops / time_mean
        return throughput
        

        

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
    