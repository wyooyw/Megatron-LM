import torch
from torch._subclasses.fake_tensor import FakeTensor
import types
from megatron.core.extensions.wyo.graph.utils import print_rank_0, is_rank_0
from megatron.core.extensions.wyo.graph.graph_utils import get_alias_union_set, get_fake_tensor
from megatron.core.extensions.wyo.model.submodule.flash_attn import flash_attn_inplace
from megatron.core.extensions.wyo.model.operator.te_layernorm import layer_norm_inplace
from megatron.core.extensions.wyo.model.communicate.communicate import reduce_scatter_along_first_dim_in_tp_group_inplace, gather_along_first_dim_in_tp_group_inplace
import time

def get_dtype(node0):
    # meta_val0 = node0.meta.get(
    #     "val", node0.meta.get("tensor_meta", node0.meta.get("example_value", None))
    # )
    # return meta_val0.dtype
    return get_fake_tensor(node0).dtype

def get_shape(node0):
    # meta_val0 = node0.meta.get(
    #     "val", node0.meta.get("tensor_meta", node0.meta.get("example_value", None))
    # )
    # return meta_val0.shape
    return get_fake_tensor(node0).shape

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

def get_life_time(node, node2line):
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
        node2line=None,
    ):
        self.root_node = root_node
        self.alias_nodes = set(alias_nodes)
        self.node2line = node2line
        self.meta_val = get_fake_tensor(self.root_node)
        self._infer_buffer_size()
        self._init_first_and_last_alias()

        self._can_reuse_buffer_of = []
        self._reuse_buffer_of = None

    def replace_node(self, old_node, new_node):
        if self.root_node==old_node:
            self.root_node = new_node
        assert old_node in self.alias_nodes
        assert new_node not in self.alias_nodes
        self.alias_nodes.remove(old_node)
        self.alias_nodes.add(new_node)

    def add_node(self, new_node):
        assert new_node not in self.alias_nodes
        self.alias_nodes.add(new_node)

    def add_nodes(self, new_nodes):
        for node in new_nodes:
            self.add_node(node)

    def _infer_buffer_size(self):
        if type(self.meta_val) == FakeTensor:
            self.buffer_size = (
                self.meta_val.element_size() * self.meta_val.nelement()
            )
        else:
            self.buffer_size = None

    def _init_first_and_last_alias(self):
        first_time = None
        first_node = None
        last_time = None
        last_node = None

        for node in self.alias_nodes:
            line = self.node2line[node]
            min_life_time, max_life_time = get_life_time(node, self.node2line)
            if first_time is None or min_life_time <= first_time:
                first_node = node
                first_time = min_life_time
            if last_time is None or max_life_time >= last_time:
                last_node = node
                last_time = max_life_time

        self._first_alias_node = first_node
        self._last_alias_node = last_node
        self.life_begin = first_time
        self.life_end = last_time

    def is_before_and_disjoint(self, val):
        return self.life_end < val.life_begin

    def is_same_buffer_size(self, val):
        if val.buffer_size is None or self.buffer_size is None:
            return False
        return val.buffer_size == self.buffer_size

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

def print_graph_with_line_index(graph):
    lines = f"{graph}".split("\n")
    head = lines[0]
    body = lines[1:]
    body = [f"{idx} {line}" for idx, line in enumerate(body)]
    body = [head, *body]
    print("\n".join(body))

class ValuesManager:
    def __init__(self, graph, values):
        self.graph = graph
        self.values = values

    def update_node2line(self):
        node2line = {node: line for line, node in enumerate(self.graph.nodes)}
        for value in self.values:
            value.node2line = node2line
            value._init_first_and_last_alias()

    def merge_value(self, value_formmer, value_latter, change_values = False):
        root_node = value_formmer.root_node
        alias_nodes = value_formmer.alias_nodes | value_latter.alias_nodes
        node2line = value_formmer.node2line
        new_value = Value(root_node, alias_nodes, node2line)

        if change_values:
            self.values.remove(value_formmer)
            self.values.remove(value_latter)
            self.values.append(new_value)

        return new_value

    def change_dtype_and_view_for_reuse(self, reuse_node, cur_node):
        # change dtype
        added_nodes = []
        # if not get_dtype(reuse_node) == get_dtype(cur_node):

        # view as a continuous 1d buffer
        reuse_fake_tensor = get_fake_tensor(reuse_node)
        if reuse_fake_tensor is None:
            print_rank_0(f"{reuse_node.target=}")
            exit()
        reuse_size = reuse_fake_tensor.numel()
        reuse_node = self.graph.call_function(
            torch.ops.aten.as_strided,
            (reuse_node, (reuse_size,), (1,), 0),
        )
        added_nodes.append(reuse_node)

        # change dtype
        reuse_node = self.graph.call_function(
            torch.ops.aten.view,
            (reuse_node, get_dtype(cur_node)),
        )
        added_nodes.append(reuse_node)

        # change shape
        reuse_node = self.graph.call_function(
            torch.ops.aten.view,
            (reuse_node, get_shape(cur_node)),
        )

        added_nodes.append(reuse_node)

        return added_nodes

    def replace_args_of_users(self, old_node, new_node):
        users = list(old_node.users.keys())
        for user in users:
            replace_args(user, old_node, new_node)

    def reuse_buffer(self, value_formmer, value_latter):
        success = False
        reuse_node = value_formmer._last_alias_node
        cur_node = value_latter._first_alias_node
        # print(f"try reuse: {cur_node.op=}, {type(cur_node.target)=}")
        
        reuse_pattern = {
            "mm.default": torch.ops.aten.mm.out,
            "_to_copy.default": torch.ops.aten._to_copy.out,
            "add.Tensor": torch.ops.aten.add.out,
            "clone.default": torch.ops.aten.clone.out,
            "gelu.default": torch.ops.aten.gelu.out,
            "flash_attn.default": flash_attn_inplace,
            "layer_norm.default": layer_norm_inplace,
            "gather_along_first_dim_in_tp_group.default": gather_along_first_dim_in_tp_group_inplace,
            "reduce_scatter_along_first_dim_in_tp_group.default": reduce_scatter_along_first_dim_in_tp_group_inplace,
            # "gelu_backward.default": torch.ops.aten.gelu_backward,
        }
        if (
            cur_node.op == "call_function"
            and hasattr(cur_node.target, "__name__")
            # and type(cur_node.target) == torch._ops.OpOverload
            and cur_node.target.__name__ in reuse_pattern
        ):

            # step 1: merge two value
            new_value = self.merge_value(value_formmer, value_latter, change_values=True)

            # replace to add.out
            with self.graph.inserting_before(cur_node):

                # step 2: add dtype and view op of the buffer
                added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
                new_value.add_nodes(added_nodes)
                reuse_node = added_nodes[-1]

                # step 3: replace cur_node with inplace_node
                kwargs = dict(cur_node.kwargs)
                assert "out" not in kwargs
                kwargs["out"] = reuse_node
                inplace_node = self.graph.call_function(
                    reuse_pattern[cur_node.target.__name__], cur_node.args, kwargs
                )
                inplace_node.meta["val"] = get_fake_tensor(cur_node).clone()

                self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
                self.graph.erase_node(cur_node)
                new_value.replace_node(cur_node, inplace_node)

            # step 4: update node2line
            self.update_node2line()

            success = True
            print("reuse success!")

        # if (
        #     cur_node.op == "call_function"
        #     and type(cur_node.target) == torch._ops.OpOverload
        #     and cur_node.target.__name__ == "mm.default"
        # ):
        #     # step 1: merge two value
        #     new_value = self.merge_value(value_formmer, value_latter, change_values=True)

        #     # replace to mm.out
        #     with self.graph.inserting_before(cur_node):

        #         # step 2: add dtype and view op of the buffer
        #         added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
        #         new_value.add_nodes(added_nodes)
        #         reuse_node = added_nodes[-1]

        #         # step 3: replace cur_node with inplace_node
        #         inplace_node = self.graph.call_function(
        #             torch.ops.aten.mm.out, cur_node.args, {"out": reuse_node}
        #         )
        #         self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
        #         self.graph.erase_node(cur_node)
        #         new_value.replace_node(cur_node, inplace_node)

        #     # step 4: update node2line
        #     self.update_node2line()

        #     success = True
        #     print("reuse success!")

        # elif (
        #     cur_node.op == "call_function"
        #     and type(cur_node.target) == torch._ops.OpOverload
        #     and cur_node.target.__name__ == "_to_copy.default"
        # ):
        #     # step 1: merge two value
        #     new_value = self.merge_value(value_formmer, value_latter, change_values=True)

        #     # replace to _to_copy.out
        #     with self.graph.inserting_before(cur_node):

        #         # step 2: add dtype and view op of the buffer
        #         added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
        #         new_value.add_nodes(added_nodes)
        #         reuse_node = added_nodes[-1]

        #         # step 3: replace cur_node with inplace_node
        #         inplace_node = self.graph.call_function(
        #             torch.ops.aten._to_copy.out, cur_node.args, {"out": reuse_node}
        #         )
        #         self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
        #         self.graph.erase_node(cur_node)
        #         new_value.replace_node(cur_node, inplace_node)

        #     # step 4: update node2line
        #     self.update_node2line()

        #     success = True
        #     print("reuse success!")

        # elif (
        #     cur_node.op == "call_function"
        #     and type(cur_node.target) == torch._ops.OpOverload
        #     and cur_node.target.__name__ == "add.Tensor"
        # ):

        #     # step 1: merge two value
        #     new_value = self.merge_value(value_formmer, value_latter, change_values=True)

        #     # replace to add.out
        #     with self.graph.inserting_before(cur_node):

        #         # step 2: add dtype and view op of the buffer
        #         added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
        #         new_value.add_nodes(added_nodes)
        #         reuse_node = added_nodes[-1]

        #         # step 3: replace cur_node with inplace_node
        #         inplace_node = self.graph.call_function(
        #             torch.ops.aten.add.out, cur_node.args, {"out": reuse_node}
        #         )
        #         self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
        #         self.graph.erase_node(cur_node)
        #         new_value.replace_node(cur_node, inplace_node)

        #     # step 4: update node2line
        #     self.update_node2line()

        #     success = True
        #     print("reuse success!")

        # elif (
        #     cur_node.op == "call_function"
        #     and type(cur_node.target) == torch._ops.OpOverload
        #     and cur_node.target.__name__ == "clone.default"
        # ):

        #     # step 1: merge two value
        #     new_value = self.merge_value(value_formmer, value_latter, change_values=True)

        #     # replace to add.out
        #     with self.graph.inserting_before(cur_node):

        #         # step 2: add dtype and view op of the buffer
        #         added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
        #         new_value.add_nodes(added_nodes)
        #         reuse_node = added_nodes[-1]

        #         # step 3: replace cur_node with inplace_node
        #         kwargs = dict(cur_node.kwargs)
        #         assert "out" not in kwargs
        #         kwargs["out"] = reuse_node
        #         inplace_node = self.graph.call_function(
        #             torch.ops.aten.clone.out, cur_node.args, kwargs
        #         )
        #         self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
        #         self.graph.erase_node(cur_node)
        #         new_value.replace_node(cur_node, inplace_node)

        #     # step 4: update node2line
        #     self.update_node2line()

        #     success = True
        #     print("reuse success!")
        # else:
        #     print_rank_0(f"Resue fail! {cur_node=}")
        
            
        return success
    
    # def try_reuse_buffer(self, forbid_reuse_nodes):
    #     has_change = False
    #     self.values.sort(key=lambda x: x.life_begin)
    #     total_values = len(self.values)
    #     # find two value that can reuse
    #     print_rank_0("try_reuse_buffer:")
    #     for j in range(total_values):
    #         for i in range(0,j):
    #             vi = self.values[i]
    #             vj = self.values[j]
    #             if (vj.life_begin - vi.life_end) > 200:
    #                 continue
    #             disjoint = vi.is_before_and_disjoint(vj)
    #             same_buffer_size = vi.is_same_buffer_size(vj)
    #             contain_forbid = vi.contain_one_of(forbid_reuse_nodes) or vj.contain_one_of(
    #                 forbid_reuse_nodes
    #             )
    #             if disjoint and same_buffer_size and (not contain_forbid):
    #                 # print("Merge two value:")
    #                 # print(f"old graph: ")
    #                 # print_graph_with_line_index(self.graph)
    #                 # vi.show()
    #                 # vj.show()
    #                 # print_rank_0(f"    {vi.life_end=}, {vj.life_begin=}")
    #                 if self.reuse_buffer(vi, vj):
    #                     # print(f"new graph: ")
    #                     # print_graph_with_line_index(self.graph)
    #                     has_change = True
    #                     break

    #         if has_change:
    #             break
    #     return has_change

    def try_reuse_buffer(self, forbid_reuse_nodes, placeholder_only=True):
        has_change = False

        values_sort_by_begin = [*self.values]
        values_sort_by_begin.sort(key=lambda x: x.life_begin)

        values_sort_by_end = [*self.values]
        values_sort_by_end.sort(key=lambda x: x.life_end)

        # self.values.sort(key=lambda x: x.life_end)
        total_values = len(self.values)
        # find two value that can reuse
        print_rank_0("try_reuse_buffer:")
        for j in range(len(values_sort_by_begin)):
            for i in range(len(values_sort_by_end)-1, -1, -1):
                vi = self.values[i]
                vj = self.values[j]
                if (vj.life_begin - vi.life_end) > 200:
                    continue
                if placeholder_only and not vi.root_node.op=="placeholder":
                    continue
                disjoint = vi.is_before_and_disjoint(vj)
                same_buffer_size = vi.is_same_buffer_size(vj)
                contain_forbid = vi.contain_one_of(forbid_reuse_nodes) or vj.contain_one_of(
                    forbid_reuse_nodes
                )
                if disjoint and same_buffer_size and (not contain_forbid):
                    # print("Merge two value:")
                    # print(f"old graph: ")
                    # print_graph_with_line_index(self.graph)
                    # vi.show()
                    # vj.show()
                    # print_rank_0(f"    {vi.life_end=}, {vj.life_begin=}")
                    if self.reuse_buffer(vi, vj):
                        # print(f"new graph: ")
                        # print_graph_with_line_index(self.graph)
                        has_change = True
                        break

            if has_change:
                break
        return has_change

def _get_life_time(node, node2line):
    min_life_time = node2line[node]
    max_life_time = node2line[node]
    for user in node.users.keys():
        line = node2line[user]
        min_life_time = min(min_life_time, line)
        max_life_time = max(max_life_time, line)
    return min_life_time, max_life_time

def try_reuse_buffer_until_no_change(graph, forbid_reuse_nodes):
    begin_time = time.time()

    # get node2line
    node2line = {node: line for line, node in enumerate(graph.nodes)}

    # get all ssa values
    alias_union_set = get_alias_union_set(graph)

    # get each value's life time
    values = []
    for root, alias_values in alias_union_set.get_all_alias_values():
        value = Value(root, alias_values, node2line)
        values.append(value)

    # for value in values:
    #     print_rank_0(f"{value.root_node=}")
    #     for node in value.alias_nodes:
    #         if hasattr(node.target, "__name__"):
    #             print_rank_0(f"    alias {node.op=}, {node.target=}, {node.target.__name__=}")
    #         else:
    #             print_rank_0(f"    alias {node.op=}, {node.target=}")
    # exit()
    values_manager = ValuesManager(graph, values)
    i = 0
    while values_manager.try_reuse_buffer(forbid_reuse_nodes): 
        print(f"call try_reuse_buffer: {i}")
        i += 1
    # print(f"reuse count: {i}")
    while values_manager.try_reuse_buffer(forbid_reuse_nodes, placeholder_only=False): 
        print(f"call try_reuse_buffer: {i}")
        i += 1
    
    end_time = time.time()
    print_rank_0(f"reuse count: {i}")
    print_rank_0(f"reuse time: {(end_time-begin_time):.3f} s")