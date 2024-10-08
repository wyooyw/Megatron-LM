import os
import types
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import aot_module_simplified
from torch._subclasses.fake_tensor import FakeTensor

# from wrapped_ops.dist import all_reduce, async_all_reduce, wait
from megatron.core.extensions.wyo.model.operator.weight_update import grad_update
# from wrapped_ops.multi_out import mul_and_add
# from wrapped_ops.flash_attn import flash_attn
import numpy as np

def merge_naive(graph0, graph1):
    assert graph0 is not None
    assert graph1 is not None
    new_graph = torch.fx.graph.Graph()
    value_remap = {}
    output_0 = None
    output_1 = None

    for node in graph0.nodes:
        # print(f"{node.op=}, {node.args=}")
        if node.target == "output":
            output_0 = node
        else:
            value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])
            if value_remap[node].op == "placeholder":
                value_remap[node].target = value_remap[node].name

    for node in graph1.nodes:
        if node.target == "output":
            output_1 = node
        else:
            value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])
            if value_remap[node].op == "placeholder":
                value_remap[node].target = value_remap[node].name

    # print(f"{output_0.args[0]=}")
    # print(f"{output_1.args[0]=}")
    # print(f"{(output_0.args[0]==output_1.args[0])=}")
    args = [*output_0.args[0], *output_1.args[0]]
    remap_args = []
    for arg in args:
        if arg is None:
            remap_args.append(None)
        else:
            remap_args.append(value_remap[arg])

    new_graph.output(remap_args)

    return new_graph


def merge_overlap_comm_greedy(graph0, graph1):
    """
    args:
        graph0: compute -> communicate -> compute -> communicate
        graph1: compute -> communicate -> compute -> communicate

    returns:
        fused_graph: compute(0) -> communicate(0) ->    compute(0)  -> communicate(0)
                                       compute(1) -> communicate(1) -> compute(1)      -> communicate(1)
    """
    i0 = 0
    i1 = 0
    nodes0 = list(graph0.nodes)
    nodes1 = list(graph1.nodes)
    n_nodes_0 = len(nodes0)
    n_nodes_1 = len(nodes1)

    fused_graph = torch.fx.graph.Graph()
    value_remap = {}

    def _is_compute(node):
        return (not _is_async_comm(node)) and (
            not _is_wait(node) and (not _is_output(node))
        )

    def _is_async_comm(node):
        if node.op == "call_function":
            target = node.target
            fn_name = getattr(target, "__name__")
            return fn_name in ["async_all_reduce"]
        return False

    def _is_wait(node):
        if node.op == "call_function":
            target = node.target
            fn_name = getattr(target, "__name__")
            return fn_name == "wait"
        return False

    def _is_output(node):
        return node.target == "output"

    def _add_node_to(node, graph, value_remap):
        assert not _is_output(node), "Node is output, should not pass in this function!"
        value_remap[node] = graph.node_copy(node, lambda n: value_remap[n])
        if value_remap[node].op == "placeholder":
            value_remap[node].target = value_remap[node].name

    while i0 < n_nodes_0 - 1 and i1 < n_nodes_1 - 1:
        if (_is_compute(nodes0[i0])) and (_is_compute(nodes1[i1])):
            # add graph 0's node to fused_graph until meet comm
            while _is_compute(nodes0[i0]):
                _add_node_to(nodes0[i0], fused_graph, value_remap)
                i0 += 1

        elif _is_compute(nodes1[i1]):
            # overlap graph0's communication with graph1's computation, greedy

            assert _is_async_comm(nodes0[i0])
            assert _is_wait(nodes0[i0 + 1])

            _add_node_to(nodes0[i0], fused_graph, value_remap)
            while _is_compute(nodes1[i1]):
                _add_node_to(nodes1[i1], fused_graph, value_remap)
                i1 += 1
            _add_node_to(nodes0[i0 + 1], fused_graph, value_remap)
            i0 = i0 + 2

        elif _is_compute(nodes0[i0]):
            # overlap graph1's communication with graph0's computation, greedy

            assert _is_async_comm(nodes1[i1])
            assert _is_wait(nodes1[i1 + 1])

            _add_node_to(nodes1[i1], fused_graph, value_remap)
            while _is_compute(nodes0[i0]):
                _add_node_to(nodes0[i0], fused_graph, value_remap)
                i0 += 1
            _add_node_to(nodes1[i1 + 1], fused_graph, value_remap)
            i1 = i1 + 2

        else:
            # Both are communication. This should not happen?
            assert False, "Not implement yet."

    # Two case here: i0 point to output0, or i1 point to output 1
    if i0 == (n_nodes_0 - 1):
        assert _is_output(nodes0[i0])
        while i1 < (n_nodes_1 - 1):
            _add_node_to(nodes1[i1], fused_graph, value_remap)
            i1 += 1
        assert i1 == (n_nodes_1 - 1) and _is_output(nodes1[i1])

    elif i1 == n_nodes_1 - 1:
        assert _is_output(nodes1[i1])
        while i0 < (n_nodes_0 - 1):
            _add_node_to(nodes0[i0], fused_graph, value_remap)
            i0 += 1
        assert i0 == (n_nodes_0 - 1) and _is_output(nodes0[i0])

    # build output node
    output_0 = nodes0[i0]
    output_1 = nodes1[i1]
    assert _is_output(output_0) and _is_output(output_1)
    args = [*output_0.args[0], *output_1.args[0]]
    remap_args = []
    for arg in args:
        if arg is None:
            remap_args.append(None)
        else:
            remap_args.append(value_remap[arg])
    fused_graph.output(remap_args)

    return fused_graph


def get_dtype(node0):
    meta_val0 = node0.meta.get(
        "val", node0.meta.get("tensor_meta", node0.meta.get("example_value", None))
    )
    return meta_val0.dtype


def get_shape(node0):
    meta_val0 = node0.meta.get(
        "val", node0.meta.get("tensor_meta", node0.meta.get("example_value", None))
    )
    return meta_val0.shape


def get_alias_union_set(graph):
    class UnionSet:
        def __init__(self, ele_list):
            self.ele_list = ele_list
            assert len(set(self.ele_list)) == len(self.ele_list)
            self._init_ele_to_root()

        def _init_ele_to_root(self):
            self._ele_to_root = {}
            for ele in self.ele_list:
                self._ele_to_root[ele] = ele

        def get_root(self, ele):
            root = self._ele_to_root[ele]
            if root == ele:
                return root
            else:
                root = self.get_root(root)
                self._ele_to_root[ele] = root
                return root

        def merge(self, a, b):
            root_a = self.get_root(a)
            root_b = self.get_root(b)
            self._ele_to_root[root_a] = root_b

        def get_all_alias_values(self):
            all_alias_values = {}
            for ele in self.ele_list:
                root = self.get_root(ele)
                if root in all_alias_values:
                    all_alias_values[root].append(ele)
                else:
                    all_alias_values[root] = [ele]
            return all_alias_values.items()

    _not_alias_node = set()

    def _is_alias_node(node):
        """
        use black list or white list?

        return:
            is_alias_node: true / false
            alias_arg_index: int
        """
        if node.op == "call_function":
            if type(node.target) == torch._ops.OpOverload:
                if node.target.__name__ in ["t.default", "detach.default", "view.default"]:
                    return True, 0
            elif isinstance(node.target, types.FunctionType):
                if node.target.__name__ in ["async_all_reduce", "wait", "grad_update"]:
                    return True, 0

        _not_alias_node.add(
            (node.op, node.target)
        )  # print it at the end, let user easy to check
        return False, None

    # get all ssa values
    values = [node for node in graph.nodes]
    alias_union_set = UnionSet(values)
    for node in graph.nodes:
        is_alias_node, alias_arg_index = _is_alias_node(node)
        if is_alias_node:
            alias_union_set.merge(node, node.args[alias_arg_index])

    return alias_union_set


def buffer_reuse(graph, forbid_reuse_nodes):
    """
    torch.ops.aten.mm.default -> torch.ops.aten.mm.out
    """
    print("forbid_reuse_nodes:")
    for node in forbid_reuse_nodes:
        print(f"    {node}")

    def _get_life_time(node, node2line):
        min_life_time = node2line[node]
        max_life_time = node2line[node]
        for user in node.users.keys():
            line = node2line[user]
            min_life_time = min(min_life_time, line)
            max_life_time = max(max_life_time, line)
        return min_life_time, max_life_time

    class Value:
        def __init__(
            self,
            root_node=None,
            alias_nodes=None,
            life_begin=None,
            life_end=None,
            node2line=None,
        ):
            self.root_node = root_node
            self.alias_nodes = alias_nodes
            self.life_begin = life_begin
            self.life_end = life_end
            self.node2line = node2line
            self.meta_val = root_node.meta.get(
                "val",
                root_node.meta.get(
                    "tensor_meta", root_node.meta.get("example_value", None)
                ),
            )
            self._infer_buffer_size()
            self._init_first_and_last_alias()

            self._can_reuse_buffer_of = []
            self._reuse_buffer_of = None

        def _infer_buffer_size(self):
            if type(self.meta_val) == FakeTensor:
                self.buffer_size = (
                    self.meta_val.element_size() * self.meta_val.nelement()
                )
            else:
                self.buffer_size = None

        def _init_first_and_last_alias(self):
            first_pos = None
            first_node = None
            last_pos = None
            last_node = None

            for node in self.alias_nodes:
                line = self.node2line[node]
                if first_pos is None or line <= first_pos:
                    first_node = node
                    first_pos = line
                if last_pos is None or line >= last_pos:
                    last_node = node
                    last_pos = line

            self._first_alias_node = first_node
            self._last_alias_node = last_node

        def is_before_and_disjoint(self, val):
            return self.life_end < val.life_begin

        def is_same_buffer_size(self, val):
            if val.buffer_size is None or self.buffer_size is None:
                return False
            return val.buffer_size == self.buffer_size

        def can_reuse_buffer_of(self, val):
            assert not val == self
            assert val.is_before_and_disjoint(self)
            self._can_reuse_buffer_of.append(val)

        def try_reuse_buffer(self):
            if self._reuse_buffer_of is None:
                return
            reuse_node = self._reuse_buffer_of._last_alias_node
            cur_node = self._first_alias_node
            print(f"try reuse: {cur_node.op=}, {type(cur_node.target)=}")
            if (
                cur_node.op == "call_function"
                and type(cur_node.target) == torch._ops.OpOverload
                and cur_node.target.__name__ == "mm.default"
            ):
                # replace to mm.out
                with graph.inserting_before(cur_node):
                    # check dtype
                    if not get_dtype(reuse_node) == get_dtype(cur_node):
                        reuse_node = graph.call_function(
                            torch.ops.aten.view,
                            (reuse_node, get_dtype(cur_node)),
                        )
                    # if not get_shape(reuse_node) == get_shape(cur_node):
                    reuse_node = graph.call_function(
                        torch.ops.aten.resize_,
                        (reuse_node, (0,)),
                    )
                    inplace_node = graph.call_function(
                        torch.ops.aten.mm.out, cur_node.args, {"out": reuse_node}
                    )
                    # replace
                    # TODO: also replace kwargs
                    users = list(cur_node.users.keys())
                    for user in users:
                        replace_args(user, cur_node, inplace_node)
                    graph.erase_node(cur_node)
                    print("reuse success!")

        def contain_one_of(self, nodes):
            return len(set(nodes) & set(self.alias_nodes)) > 0

        def show(self):
            print(f"\nvalue {self.root_node.name}:")
            alias_names = [node.name for node in self.alias_nodes]
            alias_names = ",".join(alias_names)
            print(f"    alias: {alias_names}")
            print(f"    life_time: [ {self.life_begin} , {self.life_end} ]")
            print(f"    buffer_size: {self.buffer_size}")

            can_reuse_buffer_name = [
                value.root_node.name for value in self._can_reuse_buffer_of
            ]
            can_reuse_buffer_name = ",".join(can_reuse_buffer_name)
            print(f"    can_reuse_buffer: {can_reuse_buffer_name}")
            print(f"    reuse_buffer: {self._reuse_buffer_of}")

    # get node2line
    node2line = {node: line for line, node in enumerate(graph.nodes)}

    # get all ssa values
    alias_union_set = get_alias_union_set(graph)

    # get each value's life time
    # root_life_time = {}
    values = []
    for root, alias_values in alias_union_set.get_all_alias_values():
        min_life_begin_time = None
        max_life_end_time = None
        for alias in alias_values:
            life_begin_time, life_end_time = _get_life_time(alias, node2line)
            min_life_begin_time = (
                life_begin_time
                if min_life_begin_time is None
                else min(min_life_begin_time, life_begin_time)
            )
            max_life_end_time = (
                life_end_time
                if max_life_end_time is None
                else max(max_life_end_time, life_end_time)
            )
        # assert root not in root_life_time
        # root_life_time[root] = (min_life_begin_time, max_life_end_time)
        value = Value(
            root, alias_values, min_life_begin_time, max_life_end_time, node2line
        )
        values.append(value)

    # find disjoint values
    # sort by begin time
    values.sort(key=lambda x: x.life_begin)
    for i in range(0, len(values)):
        for j in range(i + 1, len(values)):
            vi = values[i]
            vj = values[j]
            assert not vi == vj
            assert vi.life_begin <= vj.life_end

            # judge if vi/vj's life time disjoint
            disjoint = vi.is_before_and_disjoint(vj)
            same_buffer_size = vi.is_same_buffer_size(vj)
            contain_forbid = vi.contain_one_of(forbid_reuse_nodes) or vj.contain_one_of(
                forbid_reuse_nodes
            )
            if disjoint and same_buffer_size and (not contain_forbid):
                vj.can_reuse_buffer_of(vi)
                print("can share memory!!")

    # greedy judge share memorys
    reused_bufs = set()
    for value in values:
        for reuse_buf in value._can_reuse_buffer_of:
            if reuse_buf not in reused_bufs:
                value._reuse_buffer_of = reuse_buf
                reused_bufs.add(reuse_buf)
                break

    # share memorys
    for value in values:
        value.try_reuse_buffer()

    # print(graph)
    # for value in values:
    #     value.show()



def find_inputs_alias_in_outputs(graph):
    alias_union_set = get_alias_union_set(graph)
    args_root_set = set()
    for node in graph.nodes:
        if node.op == "placeholder":
            root = alias_union_set.get_root(node)
            args_root_set.add(root)

    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break

    assert output_node is not None
    inputs_alias_in_outputs = []
    inputs_alias_in_outputs_bitset = []
    for node in output_node.args[0]:
        root = alias_union_set.get_root(node)
        if root in args_root_set:
            inputs_alias_in_outputs.append(node)
            inputs_alias_in_outputs_bitset.append(1)
        else:
            inputs_alias_in_outputs_bitset.append(0)
    inputs_alias_in_outputs_bitset = np.array(inputs_alias_in_outputs_bitset, dtype=bool)
    return inputs_alias_in_outputs, inputs_alias_in_outputs_bitset

def find_alias_in_outputs(graph):
    alias_union_set = get_alias_union_set(graph)
    output_node = get_output_node(graph)
    record = set()
    alias_in_outputs_bitset = []
    for node in output_node.args[0]:
        root = alias_union_set.get_root(node)
        if root in record:
            alias_in_outputs_bitset.append(1)
        else:
            record.add(root)
            alias_in_outputs_bitset.append(0)
    return np.array(alias_in_outputs_bitset, dtype=bool)

def get_output_node(graph):
    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break
    return output_node


def get_last_placeholder(graph):
    last_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            last_placeholder = node
    return last_placeholder


def param_grad_update(backward_graph, param_grad_index_in_output: list):
    # get all weight_grads define op
    output_node = get_output_node(backward_graph)
    last_placeholder = get_last_placeholder(backward_graph)
    output_args = list(output_node.args[0])
    for param_idx, param_pos in enumerate(param_grad_index_in_output):
        assert type(param_pos) == int
        param_grads_define_node = output_args[param_pos]
        with backward_graph.inserting_after(last_placeholder):
            placeholder_node = backward_graph.placeholder(f"param_grads_{param_idx}")
            last_placeholder = placeholder_node
        with backward_graph.inserting_after(param_grads_define_node):
            grad_update_node = backward_graph.call_function(
                grad_update, (placeholder_node, param_grads_define_node)
            )
            output_args[param_pos] = grad_update_node

    output_node.args = (output_args,)


def replace_args(node, old_arg, new_arg):
    args = node.args
    if node.op == "output":
        assert len(args) == 1, f"{args=}"
        args = args[0]

    new_args = []
    for arg in args:
        if arg == old_arg:
            new_args.append(new_arg)
        else:
            new_args.append(arg)

    if node.op == "output":
        new_args = (new_args,)
    node.args = tuple(new_args)


def allreduce_sync_to_async(graph):
    allreduce_nodes = []
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "op")
            and node.target._opname == "all_reduce"
        ):
            allreduce_nodes.append(node)

    for node in allreduce_nodes:
        # create new nodes
        with graph.inserting_before(node):
            node_async_allreduce = graph.call_function(async_all_reduce, node.args)
            node_wait = graph.call_function(wait, args=(node_async_allreduce,))
        target_fn = node_async_allreduce.target

        # replace
        users = list(node.users.keys())
        for user in users:
            replace_args(user, node, node_wait)
        assert len(node.users) == 0, f"{node.users=}"
        assert len(node_wait.users) == len(
            users
        ), f"{len(node_wait.users)=}, {len(users)=}"
        graph.erase_node(node)


def dump_code(code, template_path, output_path):

    import os

    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    src_folder, src_file = os.path.split(template_path)

    # 创建 Jinja2 环境和加载器
    env = Environment(loader=FileSystemLoader(src_folder), undefined=StrictUndefined)

    # 加载模板
    template = env.get_template(src_file)

    context = {"code": code}

    # 渲染模板
    output = template.render(context)

    with open(output_path, "w") as f:
        f.write(output)


def dist_init():
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK')}")
    torch.cuda.set_device(device)


# def print_rank_0(text):
#     if torch.distributed.get_rank()==0:
#         print(text)
class Model(nn.Module):
    def __init__(self, batch, seqlen, hidden, head):
        super().__init__()
        self.batch = batch
        self.seqlen = seqlen
        self.hidden = hidden
        self.head = head
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(q.shape[0], q.shape[1], self.head, q.shape[2] // self.head)
        k = k.reshape(k.shape[0], k.shape[1], self.head, k.shape[2] // self.head)
        v = v.reshape(v.shape[0], v.shape[1], self.head, v.shape[2] // self.head)
        q = q.bfloat16()
        k = k.bfloat16()
        v = v.bfloat16()
        o,_,_ = flash_attn(q, k, v, 0.0, 1, True, False, True)
        o = o.float()
        o = o.reshape(o.shape[0], o.shape[1], -1)
        y = self.o_proj(o)
        
        return y


class GraphKeeper:
    def __init__(self):
        self.forward_graph = None
        self.backward_graph = None
        self.params_flat = None

    def add_graph(self, graph):
        if self.forward_graph is None:
            print("add forward graph")
            self.forward_graph = graph
        elif self.backward_graph is None:
            print("add backward graph")
            self.backward_graph = graph
        else:
            assert False


def graph_capture_backend(gm, sample_inputs, keeper):
    def my_compiler(gm, sample_inputs):
        print("AOTAutograd produced a fx Graph in Aten IR")
        keeper.add_graph(gm.graph)
        # gm.print_readable()
        # verbose_python_code = gm.graph.python_code(
        #     root_module="self", verbose=True
        # )
        # print(f"{verbose_python_code.src}")
        # exit()
        return gm.forward

    params = {
        **dict(gm.named_parameters(remove_duplicate=False)),
        **dict(gm.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    keeper.params_flat = params_flat
    assert len(keeper.params_flat) > 0
    
    # Invoke AOTAutograd
    return aot_module_simplified(gm, sample_inputs, fw_compiler=my_compiler)


def test_forward_and_backward():

    batch = 4
    seqlen = 64
    hidden = 128
    head = 4
    model = Model(batch, seqlen, hidden, head).cuda()
    input = torch.randn(batch, seqlen, hidden).cuda()
    out1 = model(input)
    out1.sum().backward()
    q_proj_grad = model.q_proj.weight.grad.detach().clone()
    k_proj_grad = model.k_proj.weight.grad.detach().clone()
    # proj2_grad = model.proj2.weight.grad.detach().clone()

    # group = torch.distributed.new_group([0,1])
    # print(f"{type(group)=}")
    torch._dynamo.reset()
    keeper = GraphKeeper()
    fn = torch.compile(
        backend=partial(graph_capture_backend, keeper=keeper),
        dynamic=False,
        fullgraph=True,
    )(model)

    # triggers compilation of forward graph on the first run
    out = fn(input)
    out.sum().backward()

    print("\n--------------------------- forward --------------------------\n")
    print(keeper.forward_graph)

    print("\n--------------------------- backward --------------------------\n")
    print(keeper.backward_graph)
    # return
    # return
    # allreduce_sync_to_async(keeper.forward_graph)
    # print("\n--------------------------- async allreduce forward --------------------------\n")
    # print(keeper.forward_graph)

    # python_code = keeper.forward_graph.python_code("RootModule")
    # src = python_code.src
    # print("\n--------------------------- python code --------------------------\n")
    # print(src)
    # print(python_code.globals)
    print("\n--------------------------- run forward --------------------------\n")
    python_code = keeper.forward_graph.python_code("RootModule")
    code = python_code.src
    # dump_code(code, "template/template.jinja", "template/forward.py")
    # with open("template/forward.py", 'r') as file:
    #     code = file.read()

    local_namespace = {}
    exec(code, python_code.globals, local_namespace)
    forward = local_namespace["forward"]
    outs = forward(None, model.q_proj.weight, model.k_proj.weight, model.v_proj.weight, model.o_proj.weight, input)
    print(f"{out1=}")
    print(f"{outs[0]=}")
    # return
    # for out in outs:
    #     print(f"{out.shape=}")

    print("\n--------------------------- run backward --------------------------\n")
    python_code = keeper.backward_graph.python_code("BackwardModule")
    code = python_code.src
    dump_code(code, "template/template.jinja", "template/backward.py")
    with open("template/backward.py", "r") as file:
        code = file.read()

    local_namespace = {}
    exec(code, python_code.globals, local_namespace)
    backward = local_namespace["forward"]

    for out in outs:
        print(f"{out.shape=}")
    grad_output = torch.ones(out1.shape, dtype=out1.dtype).cuda()
    print(f"{grad_output.shape=}")
    back_outs = backward(None, *outs[1:], grad_output)
    rank = 0 #torch.distributed.get_rank()
    print(
        f"\n---------------------------\n{rank=}\n{back_outs[0]=}\n{q_proj_grad=}\n---------------------------\n"
    )
    # print(f"{proj_grad=}")
    print(
        f"\n---------------------------\n{rank=}\n{back_outs[1]=}\n{k_proj_grad=}\n---------------------------\n"
    )
    return
    # print(f"{back_outs[1]=}")
    # print(f"{proj2_grad=}")

    # for node in keeper.forward_graph.nodes:
    #     print(node.target, type(node.target))
    #     if node.target=="output":
    #         args = node.args
    #         print(args)
    #         print(type(args))

    # print("\n--------------------------- backward -------------------------\n")
    # print(keeper.backward_graph)

    # print("\n--------------------------- forward & backward -------------------------\n")
    new_graph = merge_naive(keeper.forward_graph, keeper.backward_graph)
    # print(new_graph)

    # python_code = new_graph.python_code("RootModule")
    # src = python_code.src
    # print(src)


def test_fused_forward_and_backward():

    hidden = 8
    model = Model(hidden).cuda()
    input = torch.randn(4, hidden).cuda()
    out1 = model(input)
    out1.sum().backward()
    proj_grad = model.proj.weight.grad.detach().clone()
    proj2_grad = model.proj2.weight.grad.detach().clone()

    params = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    print(type(params_flat))
    print(type(params_flat[0]))
    print(params_flat[0])
    return

    torch._dynamo.reset()
    fn = torch.compile(backend=partial(graph_capture_backend), dynamic=False)(model)

    # triggers compilation of forward graph on the first run
    out = fn(input)
    out.sum().backward()

    print("\n--------------------------- forward --------------------------\n")
    print(keeper.forward_graph)

    print("\n--------------------------- backward --------------------------\n")
    print(keeper.backward_graph)

    print("\n--------------------------- run fwd&fwd --------------------------\n")
    copied_fwd_graph = torch.fx.graph.Graph()
    rv = copied_fwd_graph.graph_copy(keeper.forward_graph, {})
    copied_fwd_graph.output(rv)
    print("copied graph:")
    print(copied_fwd_graph)

    new_graph = merge_naive(keeper.forward_graph, copied_fwd_graph)
    print(new_graph)
    python_code = new_graph.python_code("RootModule")
    code = python_code.src
    dump_code(code, "template/template.jinja", "template/forward_forward.py")
    with open("template/forward_forward.py", "r") as file:
        code = file.read()
    # return
    local_namespace = {}
    exec(code, python_code.globals, local_namespace)
    forward_forward = local_namespace["forward"]
    outs = forward_forward(
        None,
        model.proj.weight,
        model.proj2.weight,
        input,
        model.proj.weight,
        model.proj2.weight,
        input,
    )

    print(f"{out1=}")
    print(f"{outs[0]=}")
    print(f"{outs[4]=}")
    # return
    # for out in outs:
    #     print(f"{out.shape=}")

    print("\n--------------------------- run backward --------------------------\n")
    python_code = keeper.backward_graph.python_code("BackwardModule")
    code = python_code.src
    dump_code(code, "template/template.jinja", "template/backward.py")
    with open("template/backward.py", "r") as file:
        code = file.read()

    local_namespace = {}
    exec(code, python_code.globals, local_namespace)
    backward = local_namespace["forward"]

    for out in outs:
        print(f"{out.shape=}")
    grad_output = torch.ones(out1.shape, dtype=out1.dtype).cuda()
    print(f"{grad_output.shape=}")
    back_outs = backward(None, *outs[1:], grad_output)
    rank = 0#torch.distributed.get_rank()
    print(
        f"\n---------------------------\n{rank=}\n{back_outs[0]=}\n{q_proj_grad=}\n---------------------------\n"
    )
    # print(f"{proj_grad=}")
    print(
        f"\n---------------------------\n{rank=}\n{back_outs[1]=}\n{k_proj_grad=}\n---------------------------\n"
    )
    # print(f"{back_outs[1]=}")
    # print(f"{proj2_grad=}")


if __name__ == "__main__":
    # dist_init()
    test_forward_and_backward()
    # dist.destroy_process_group()
