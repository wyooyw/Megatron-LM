import gc
import os
import random
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed
import torch.nn as nn
import torch.utils._pytree as pytree
from megatron.core.extensions.wyo.status import set_current_status_run

from megatron.core.extensions.wyo.graph.args_manager import (
    BackwardForwardInputs,
    BackwardInputs,
    ModelArgsAndReturnsManager,
)
from megatron.core.extensions.wyo.graph.graph_utils import (
    GraphKeeper,
    allreduce_sync_to_async,
    buffer_reuse,
    find_inputs_alias_in_outputs,
    find_alias_in_outputs,
    graph_capture_backend,
    merge_naive,
    merge_overlap_comm_greedy,
    param_grad_update,
)
from megatron.core.extensions.wyo.graph.passes.unwrap import unwrap
from megatron.core.extensions.wyo.graph.passes.overlap_schedule import overlap_schedule
from megatron.core.extensions.wyo.graph.passes.comm_sync_to_async import comm_sync_to_async
from megatron.core.extensions.wyo.graph.utils import is_rank_0, print_graph_rank_0, print_rank_0, seed_everything
# from wrapped_ops.dist import all_reduce
# from simple_models import Attention, MultiAttentionModel, SimpleModel
# from passes.buffer_reuse import try_reuse_buffer_until_no_change
# from passes.replace_add import replace_add

class GARunner:
    def __init__(self, model, n_ga, example_args, example_kwargs, n_forward_input, n_forward_output):
        self.n_ga = n_ga
        self.model = model
        self.n_forward_input = n_forward_input
        self.n_forward_output = n_forward_output

        self.example_args = example_args
        self.example_kwargs = example_kwargs

        # self._make_params()
        self._make_graphs()
        self.args_rets_manager = ModelArgsAndReturnsManager(
            self.model, self.params_flat, self.n_forward_input, self.n_forward_output
        )

        rank = torch.distributed.get_rank()
        self.forward_fn = self._make_run_functions(
            self.forward_graph, "forward_fnnnn", f".function/forward_fn_{rank}.py"
        )
        self.backward_fn = self._make_run_functions(
            self.backward_graph, "backward_fnnnn", f".function/backward_fn_{rank}.py"
        )
        self.fused_backward_forward_fn = self._make_run_functions(
            self.fused_bwd_fwd_graph,
            "fused_backward_forward_fn",
            f".function/fused_backward_forward_fn_{rank}.py",
        )
        # exit()

    def _cnt_graph_placeholder(self, graph):
        return sum([1 for node in graph.nodes if node.op == "placeholder"])

    def _get_forbid_reuse_nodes(self):
        """
        weights and inputs can not be reuse
        """
        n_fwd_args = self._cnt_graph_placeholder(self.forward_graph)
        n_bwd_args = self._cnt_graph_placeholder(self.backward_graph)
        fwd_args_begin = n_bwd_args
        fwd_args_end = n_bwd_args + n_fwd_args  # not include

        _, bitset1 = find_inputs_alias_in_outputs(self.forward_graph)
        bitset2 = find_alias_in_outputs(self.forward_graph)
        bitset = np.logical_or(bitset1, bitset2)
        print(f"{bitset1=}")
        print(f"{bitset2=}")
        print(f"{bitset=}")
        bitset = bitset[self.n_forward_output :]

        placeholder_cnt = 0
        forbid_reuse_nodes = []
        for node in self.fused_bwd_fwd_graph.nodes:
            if node.op == "placeholder":
                if fwd_args_begin <= placeholder_cnt and placeholder_cnt < fwd_args_end:
                    forbid_reuse_nodes.append(node)
                elif placeholder_cnt < len(bitset) and bitset[placeholder_cnt] == True:
                    forbid_reuse_nodes.append(node)
                placeholder_cnt += 1

        return forbid_reuse_nodes

    def _make_graphs(self):
        keeper = GraphKeeper()
        fn = torch.compile(
            backend=partial(graph_capture_backend, keeper=keeper),
            dynamic=False,
            fullgraph=True,
        )(self.model)

        # run to capture forward graph
        out = fn(*self.example_args, **self.example_kwargs)

        # run to capture backward graph
        out.sum().backward()

        self.params_flat = keeper.params_flat
        assert len(self.params_flat) > 0

        # clear grads
        for param in self.params_flat:
            param.grad = None

        self.forward_graph = keeper.forward_graph
        self.backward_graph = keeper.backward_graph
        # for node in self.forward_graph.nodes:
        #     print_rank_0(f"{node.op=}, {node.target=}, {node.args=}")
        # exit()
        # print_rank_0(f"before: \n{self.forward_graph}\n")
        # comm_sync_to_async(self.forward_graph)
        # print_rank_0(f"after: \n{self.forward_graph}\n")
        # comm_sync_to_async(self.backward_graph)
        param_grad_update(self.backward_graph, list(range(len(self.params_flat))))
        # print_rank_0(f"\nafter param_grad_update: \n{self.backward_graph}\n")
        # for node in self.forward_graph.nodes:
        #     name = getattr(node.target, "__name__") if hasattr(node.target, "__name__") else ""
        #     print_rank_0(f"{node.op=}, {node.target=}, {name=}")
        # exit()
        self.fused_bwd_fwd_graph = merge_naive(self.backward_graph, self.forward_graph)
        # self.fused_bwd_fwd_graph = merge_overlap_comm_greedy(
        #     self.backward_graph, self.forward_graph
        # )
        self.fused_bwd_fwd_graph = overlap_schedule(self.fused_bwd_fwd_graph)
        unwrap(self.fused_bwd_fwd_graph)
        # print_rank_0("naive fused graph: ")
        # print_graph_rank_0(self.fused_bwd_fwd_graph)
        # self.fused_bwd_fwd_graph = merge_overlap_comm_greedy(
        #     self.backward_graph, self.forward_graph
        # )
        # replace_add(self.fused_bwd_fwd_graph)
        # print_rank_0(f"\nafter replace_add:")
        # print_graph_rank_0(self.fused_bwd_fwd_graph)
        # exit()

        # reuse buffer
        # print_rank_0("Before buffer reuse:")
        # print_graph_rank_0(self.fused_bwd_fwd_graph)
        # forbid_reuse_nodes = self._get_forbid_reuse_nodes()
        # try_reuse_buffer_until_no_change(self.fused_bwd_fwd_graph, forbid_reuse_nodes)
        # buffer_reuse(self.fused_bwd_fwd_graph, forbid_reuse_nodes)
        # print_rank_0("After buffer reuse:")
        # print_graph_rank_0(self.fused_bwd_fwd_graph)
        # exit()

    def _make_params(self):
        params = {
            **dict(self.model.named_parameters(remove_duplicate=False)),
            **dict(self.model.named_buffers(remove_duplicate=False)),
        }
        params_flat, params_spec = pytree.tree_flatten(params)
        params_flat = list(params_flat)
        self.params_flat = params_flat

    def _make_run_functions(self, graph, name, dump_path=None):
        python_code = graph.python_code(name, verbose=True)
        code = python_code.src
        if dump_path is not None:
            with open(dump_path, "w") as f:
                f.write(code)
        # return
        local_namespace = {}
        exec(code, python_code.globals, local_namespace)
        fn = local_namespace["forward"]
        return fn

    # def update_param_grad(self, param_grads):
    #     assert len(self.params_flat) <= len(
    #         param_grads
    #     ), f"{len(self.params_flat)=}, {len(param_grads)=}"
    #     assert len(param_grads)==len(self.params_flat)
    #     # param_grads = param_grads[: len(self.params_flat)]
    #     for param, grad in zip(self.params_flat, param_grads):
    #         if grad is None:
    #             continue
    #         if param.grad is None:
    #             param.grad = grad  # .clone()
    #         else:
    #             param.grad += grad

    def init_param_grad(self):
        # param_grads = param_grads[: len(self.params_flat)]
        for param in self.params_flat:
            if param.grad is None:
                param.grad = torch.zeros(
                    param.shape, dtype=param.dtype, device=param.device
                )

    def run(self, data_loader_fn, post_process_fn):
        # set_current_status_run()
        with torch.no_grad():
            return self._run(data_loader_fn, post_process_fn)

    def check_inputs(self, inputs):
        print("Check inputs:")
        size = len(inputs)
        print("\t", end="")
        for j in range(size):
            print(f"{j}\t", end=" ")
        print("")
        for i in range(size):
            print(i, end="\t")
            for j in range(size):
                if inputs[i] is None or inputs[j] is None:
                    is_same = False
                else:
                    is_same = inputs[i].data_ptr()==inputs[j].data_ptr()
                
                print("T" if is_same else "F", end="\t")
            print("")

    def _run(self, data_loader_fn, post_process_fn):
        get_and_clear_grads(self.model)
        self.init_param_grad()
        # gc.collect()
        # torch.cuda.empty_cache()

        torch.cuda.nvtx.range_push("GARunner-run")
        collect_forward_outputs = []
        # step 1: run forward
        # print_rank_0("begin run forward")
        fwd_inputs, loss_masks = data_loader_fn()
        fwd_inputs = self.args_rets_manager.make_forward_inputs(fwd_inputs)
        torch.cuda.nvtx.range_push("forward")
        fwd_outputs = self.forward_fn(*fwd_inputs.dump())
        fwd_outputs = self.args_rets_manager.make_forward_outputs(fwd_outputs)
        collect_forward_outputs.append(
            post_process_fn(fwd_outputs.model_outputs[0].detach().clone(), loss_masks)
        )
        torch.cuda.nvtx.range_pop()

        # gc.collect()
        # torch.cuda.empty_cache()

        # step 2: run fused_bwd_fwd
        for i in range(1, self.n_ga):
            # print_rank_0("begin run backward-forward")
            fwd_inputs, loss_masks = data_loader_fn()
            fwd_inputs = self.args_rets_manager.make_forward_inputs(fwd_inputs)

            fused_bwd_fwd_inputs = self.args_rets_manager.make_backward_forward_inputs(
                fwd_outputs, fwd_inputs
            )
            del fwd_outputs, fwd_inputs
            # save_for_bwd_size = fused_bwd_fwd_inputs.backward_inputs.size_of_save_for_backwards()
            # save_for_bwd_size = save_for_bwd_size / 1024 / 1024 / 1024
            # if i==1:
            #     print_rank_0(f"{save_for_bwd_size=:.3f} GB")
            # self.check_inputs(fused_bwd_fwd_inputs.dump())
            torch.cuda.nvtx.range_push("back-forward")
            fused_backward_forward_fn = partial(self.fused_backward_forward_fn, *fused_bwd_fwd_inputs.dump())
            del fused_bwd_fwd_inputs
            # gc.collect()
            fused_bwd_fwd_outputs = fused_backward_forward_fn()
            # fused_bwd_fwd_outputs = self.fused_backward_forward_fn(
            #     *fused_bwd_fwd_inputs.dump()
            # )
            fused_bwd_fwd_outputs = (
                self.args_rets_manager.make_backward_forward_outputs(
                    fused_bwd_fwd_outputs
                )
            )
            collect_forward_outputs.append(
                post_process_fn(
                    fused_bwd_fwd_outputs.forward_outputs.model_outputs[0].detach().clone(),
                    loss_masks
                )
            )
            torch.cuda.nvtx.range_pop()

            fwd_outputs = fused_bwd_fwd_outputs.forward_outputs
            grads = fused_bwd_fwd_outputs.backward_outputs.parameters_grad

            del fused_bwd_fwd_outputs
            del grads
            # del fused_bwd_fwd_inputs
            # gc.collect()
            # torch.cuda.empty_cache()
        # print_rank_0("begin run backward")
        # step 3: run last backward
        backward_inputs = self.args_rets_manager.make_backward_inputs(fwd_outputs)
        torch.cuda.nvtx.range_push("backward")
        backward_outputs = self.backward_fn(*backward_inputs.dump())
        backward_outputs = self.args_rets_manager.make_backward_outputs(
            backward_outputs
        )
        torch.cuda.nvtx.range_pop()

        # step 4: update_param
        # no need to do it. grad update is done in backward.
        del fwd_outputs
        del backward_inputs
        del backward_outputs
        # gc.collect()
        # torch.cuda.empty_cache()
        torch.cuda.nvtx.range_pop()
        return collect_forward_outputs

class NaiveRunner:
    def __init__(self, model, n_ga):
        self.n_ga = n_ga
        self.model = model

    def run(self, input_micro_batches):
        torch.cuda.nvtx.range_push("NaiveRunner-run")
        collect_forward_outputs = []
        assert self.n_ga == len(input_micro_batches)
        for idx, inputs in enumerate(input_micro_batches):
            torch.cuda.nvtx.range_push(f"batch {idx}")
            out = self.model(*inputs)
            collect_forward_outputs.append(out.detach().clone())
            out.sum().backward()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        return collect_forward_outputs


class BatchSplitRunner:
    def __init__(self, model, n_ga, half_batch_example_inputs):
        self.n_ga = n_ga
        self.model = model
        self.n_forward_input = 1
        self.n_forward_output = 1
        self.half_batch_example_inputs = half_batch_example_inputs
        self.args_rets_manager = ModelArgsAndReturnsManager(
            self.model, self.n_forward_input, self.n_forward_output
        )
        self._make_params()
        self._make_graphs()
        self.fused_forward_forward_fn = self._make_run_functions(
            self.fused_fwd_fwd_graph,
            "fused_forward_forward_fn",
            ".function/fused_forward_forward_fn.py",
        )
        self.fused_backward_backward_fn = self._make_run_functions(
            self.fused_bwd_bwd_graph,
            "fused_backward_backward_fn",
            ".function/fused_backward_backward_fn.py",
        )

    def _make_params(self):
        params = {
            **dict(self.model.named_parameters(remove_duplicate=False)),
            **dict(self.model.named_buffers(remove_duplicate=False)),
        }
        params_flat, params_spec = pytree.tree_flatten(params)
        params_flat = list(params_flat)
        self.params_flat = params_flat

    def _make_graphs(self):
        keeper = GraphKeeper()
        fn = torch.compile(
            backend=partial(graph_capture_backend, keeper=keeper),
            dynamic=False,
            fullgraph=True,
        )(self.model)

        # run to capture forward graph
        out = fn(*self.half_batch_example_inputs)

        # run to capture backward graph
        out.sum().backward()

        # clear grads
        for param in self.params_flat:
            param.grad = None

        self.forward_graph = keeper.forward_graph
        allreduce_sync_to_async(self.forward_graph)

        self.backward_graph = keeper.backward_graph
        allreduce_sync_to_async(self.backward_graph)
        param_grad_update(self.backward_graph, list(range(len(self.params_flat))))

        # fused_fwd_fwd_graph
        copied_forward_graph = torch.fx.graph.Graph()
        rv = copied_forward_graph.graph_copy(keeper.forward_graph, {})
        copied_forward_graph.output(rv)
        self.fused_fwd_fwd_graph = merge_overlap_comm_greedy(
            self.forward_graph, copied_forward_graph
        )
        print_rank_0(f"self.fused_fwd_fwd_graph:\n{self.fused_fwd_fwd_graph}")

        # fused_bwd_bwd_graph
        copied_backward_graph = torch.fx.graph.Graph()
        rv = copied_backward_graph.graph_copy(keeper.backward_graph, {})
        copied_backward_graph.output(rv)
        self.fused_bwd_bwd_graph = merge_overlap_comm_greedy(
            self.backward_graph, copied_backward_graph
        )
        print_rank_0(f"self.fused_bwd_bwd_graph:\n{self.fused_bwd_bwd_graph}")

    def _make_run_functions(self, graph, name, dump_path=None):
        python_code = graph.python_code(name, verbose=True)
        code = python_code.src
        if dump_path is not None and is_rank_0():
            with open(dump_path, "w") as f:
                f.write(code)
        # return
        local_namespace = {}
        exec(code, python_code.globals, local_namespace)
        fn = local_namespace["forward"]
        return fn

    def deal_inputs(self, inputs):
        inputs = inputs[0]
        bs = inputs.shape[0]
        assert bs % 2 == 0
        inputs_0 = inputs[: bs // 2]
        inputs_1 = inputs[bs // 2 :]
        inputs = ((inputs_0,), (inputs_1,))
        return inputs

    def init_param_grad(self):
        # param_grads = param_grads[: len(self.params_flat)]
        for param in self.params_flat:
            if param.grad is None:
                param.grad = torch.zeros(
                    param.shape, dtype=param.dtype, device=param.device
                )

    def run(self, input_micro_batches):
        with torch.no_grad():
            self._run(input_micro_batches)

    def _run(self, input_micro_batches):
        self.init_param_grad()
        torch.cuda.nvtx.range_push("BatchSplitRunner-run")

        assert self.n_ga == len(input_micro_batches)
        for inputs in input_micro_batches:
            inputs = self.deal_inputs(inputs)
            torch.cuda.nvtx.range_push("Forward & Backward")
            inputs = self.args_rets_manager.make_forward_forward_inputs(
                inputs[0], inputs[1]
            )

            torch.cuda.nvtx.range_push("Forward")
            fwd_fwd_outputs = self.fused_forward_forward_fn(*inputs.dump())
            torch.cuda.nvtx.range_pop()
            fwd_fwd_outputs = self.args_rets_manager.make_forward_forward_outputs(
                fwd_fwd_outputs
            )

            bwd_bwd_inputs = self.args_rets_manager.make_backward_backward_inputs(
                fwd_fwd_outputs
            )
            torch.cuda.nvtx.range_push("Backward")
            bwd_bwd_outputs = self.fused_backward_backward_fn(*bwd_bwd_inputs.dump())
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()


def show_memory_tensors():
    print("-------------------------- show_memory_tensors --------------------------")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass
    print("--------------------------------------------------------------------------")


def get_and_clear_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = param.grad
        param.grad = None
    return grads


def dist_init():
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK')}")
    torch.cuda.set_device(device)


def profile(fn, n_warmup, n_profile):
    for i in range(n_warmup):
        fn()

    time_list = []
    memory_list = []
    for i in range(n_profile):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        fn()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        time_list.append(elapsed_time_ms)

        max_memory = torch.cuda.max_memory_allocated()
        memory_list.append(max_memory)

    time_list = np.array(time_list)
    memory_list = np.array(memory_list)

    return (
        time_list,
        time_list.mean(),
        time_list.std(),
        memory_list,
        memory_list.mean(),
        memory_list.std(),
    )


def show_memory(title):
    gc.collect()
    torch.cuda.empty_cache()
    memory = torch.cuda.memory_allocated()
    memory = memory / 1024 / 1024 / 1024
    print(f"{title} {memory=:2f} GB")


if __name__ == "__main__":
    dist_init()
    seed_everything()

    torch.cuda.cudart().cudaProfilerStart()

    show_memory("1")
    micro_bs = 64
    seqlen = 512
    hidden = 1024
    head = 8
    n_ga = 4
    input = torch.randn(micro_bs, seqlen, hidden).cuda()
    input_micro_batches = [(torch.randn(micro_bs, seqlen, hidden).cuda(),) for i in range(n_ga)]
    # input = input_micro_batches[0][0]
    # input_micro_batches = [(torch.ones((micro_bs, hidden), dtype=torch.float32).cuda(),) for i in range(n_ga)]
    # input_micro_batches = [(torch.randn((micro_bs, hidden), dtype=torch.float32).cuda(),) for i in range(n_ga)]
    show_memory("2")
    model = SimpleModel(hidden, head).cuda()
    show_memory("3")

    # run naive runner
    naive_runner = NaiveRunner(model, n_ga)
    naive_forward_outputs = naive_runner.run(input_micro_batches)
    naive_grads = get_and_clear_grads(model)

    # run GARuner
    fused_runner = GARunner(model, n_ga=n_ga, example_inputs=(input,))

    # run BatchSplitRunner
    # split_runner = BatchSplitRunner(
    #     model, n_ga=n_ga, half_batch_example_inputs=(input[: micro_bs // 2],)
    # )
    print_rank_0("begin run fused_runner")
    test_runner = fused_runner
    test_forward_outputs = test_runner.run(input_micro_batches)
    test_grads = get_and_clear_grads(model)

    print(f"\n------------------------ Compare forward outputs ({len(test_forward_outputs)}) ------------------------\n")
    assert len(naive_forward_outputs)==len(test_forward_outputs)
    for idx, (naive_out, test_out) in enumerate(zip(naive_forward_outputs, test_forward_outputs)):
        print(f"forward output {idx}: \n    {naive_out.reshape(-1)[:10]=}\n    {test_out.reshape(-1)[:10]=}")
    print("\n------------------------------------------------------------------------\n")
    # compare result
    # assert len(naive_grads) == len(test_grads) and len(test_grads) > 0
    for idx,key in enumerate(test_grads.keys()):
        
        is_pass = torch.allclose(naive_grads[key], test_grads[key], rtol=0.01)
        # assert is_pass
        if is_pass:
            print(f"{idx} {key=} {is_pass=}")
        else:
            mean_diff = torch.mean(torch.abs(naive_grads[key] - test_grads[key]))
            max_diff = torch.max(torch.abs(naive_grads[key] - test_grads[key]))
            print(f"{idx} {key=} {is_pass=} {mean_diff=:2f} {max_diff=:2f}")
        # print(f"    {naive_grads[key].reshape(-1)[:10]=}")
        # print(f"    {test_grads[key].reshape(-1)[:10]=}")
    # dist.destroy_process_group()
    # exit()
    # Profiling
    n_warmup = 4
    n_profile = 8
    print("Begin profile!")
    print("Naive: ")
    (
        _,
        naive_mean,
        naive_std,
        naive_memory_all,
        naive_memory_mean,
        naive_memory_std,
    ) = profile(
        partial(naive_runner.run, input_micro_batches=input_micro_batches),
        n_warmup=n_warmup,
        n_profile=n_profile,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print("Experimented: ")
    (
        _,
        test_mean,
        test_std,
        test_memory_all,
        test_memory_mean,
        test_memory_std,
    ) = profile(
        partial(test_runner.run, input_micro_batches=input_micro_batches),
        n_warmup=n_warmup,
        n_profile=n_profile,
    )

    speed_up = ((naive_mean - test_mean) / naive_mean) * 100
    print(f"naive_time: {naive_mean:.3f}(±{naive_std:.3f}) ms")
    print(f"test_time: {test_mean:.3f}(±{naive_std:.3f}) ms")
    print(f"speedup: {speed_up:.2f} %")
    print(f"{naive_memory_all[0:5]/(1024 * 1024 * 1024 )=} (GB)")
    print(f"{test_memory_all[0:5]/(1024 * 1024 * 1024 )=} (GB)")

    torch.cuda.cudart().cudaProfilerStop()

    dist.destroy_process_group()
