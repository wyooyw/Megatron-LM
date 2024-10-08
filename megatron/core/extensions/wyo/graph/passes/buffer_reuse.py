import torch
from torch._subclasses.fake_tensor import FakeTensor
import types

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
        reuse_node = self.graph.call_function(
            torch.ops.aten.view,
            (reuse_node, get_dtype(cur_node)),
        )
        added_nodes.append(reuse_node)

        # change view
        reuse_node = self.graph.call_function(
            torch.ops.aten.resize_,
            (reuse_node, (0,)),
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
        if (
            cur_node.op == "call_function"
            and type(cur_node.target) == torch._ops.OpOverload
            and cur_node.target.__name__ == "mm.default"
        ):
            # step 1: merge two value
            new_value = self.merge_value(value_formmer, value_latter, change_values=True)

            # replace to mm.out
            with self.graph.inserting_before(cur_node):

                # step 2: add dtype and view op of the buffer
                added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
                new_value.add_nodes(added_nodes)
                reuse_node = added_nodes[-1]

                # step 3: replace cur_node with inplace_node
                inplace_node = self.graph.call_function(
                    torch.ops.aten.mm.out, cur_node.args, {"out": reuse_node}
                )
                self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
                self.graph.erase_node(cur_node)
                new_value.replace_node(cur_node, inplace_node)

            # step 4: update node2line
            self.update_node2line()

            success = True
            print("reuse success!")

        elif (
            cur_node.op == "call_function"
            and type(cur_node.target) == torch._ops.OpOverload
            and cur_node.target.__name__ == "_to_copy.default"
        ):
            # step 1: merge two value
            new_value = self.merge_value(value_formmer, value_latter, change_values=True)

            # replace to _to_copy.out
            with self.graph.inserting_before(cur_node):

                # step 2: add dtype and view op of the buffer
                added_nodes = self.change_dtype_and_view_for_reuse(reuse_node, cur_node)
                new_value.add_nodes(added_nodes)
                reuse_node = added_nodes[-1]

                # step 3: replace cur_node with inplace_node
                inplace_node = self.graph.call_function(
                    torch.ops.aten._to_copy.out, cur_node.args, {"out": reuse_node}
                )
                self.replace_args_of_users(cur_node, inplace_node) # TODO: also replace kwargs
                self.graph.erase_node(cur_node)
                new_value.replace_node(cur_node, inplace_node)

            # step 4: update node2line
            self.update_node2line()

            success = True
            print("reuse success!")

        
            
        return success
    
    def try_reuse_buffer(self, forbid_reuse_nodes):
        has_change = False
        self.values.sort(key=lambda x: x.life_begin)
        total_values = len(self.values)
        # find two value that can reuse
        for j in range(total_values):
            for i in range(0, j):
                vi = self.values[i]
                vj = self.values[j]
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

    # get node2line
    node2line = {node: line for line, node in enumerate(graph.nodes)}

    # get all ssa values
    alias_union_set = get_alias_union_set(graph)

    # get each value's life time
    values = []
    for root, alias_values in alias_union_set.get_all_alias_values():
        value = Value(root, alias_values, node2line)
        values.append(value)

    values_manager = ValuesManager(graph, values)
    i = 0
    while values_manager.try_reuse_buffer(forbid_reuse_nodes): 
        print(f"call try_reuse_buffer: {i}")
        i += 1
        # if i >= 6:
        #     break
        
    

        

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

    # merge by alias ops
    for node in graph.nodes:
        is_alias_node, alias_arg_index = _is_alias_node(node)
        if is_alias_node:
            alias_union_set.merge(node, node.args[alias_arg_index])
    

    return alias_union_set


# def buffer_reuse(graph, forbid_reuse_nodes):
#     """
#     torch.ops.aten.mm.default -> torch.ops.aten.mm.out
#     """
#     print("forbid_reuse_nodes:")
#     for node in forbid_reuse_nodes:
#         print(f"    {node}")

#     def _get_life_time(node, node2line):
#         min_life_time = node2line[node]
#         max_life_time = node2line[node]
#         for user in node.users.keys():
#             line = node2line[user]
#             min_life_time = min(min_life_time, line)
#             max_life_time = max(max_life_time, line)
#         return min_life_time, max_life_time

#     class Value:
#         def __init__(
#             self,
#             root_node=None,
#             alias_nodes=None,
#             life_begin=None,
#             life_end=None,
#             node2line=None,
#         ):
#             self.root_node = root_node
#             self.alias_nodes = alias_nodes
#             self.life_begin = life_begin
#             self.life_end = life_end
#             self.node2line = node2line
#             self.meta_val = root_node.meta.get(
#                 "val",
#                 root_node.meta.get(
#                     "tensor_meta", root_node.meta.get("example_value", None)
#                 ),
#             )
#             self._infer_buffer_size()
#             self._init_first_and_last_alias()

#             self._can_reuse_buffer_of = []
#             self._reuse_buffer_of = None

#         def replace_node(self, old_node, new_node):
#             if self.root_node==old_node:
#                 self.root_node = new_node
#             assert old_node in self.alias_nodes
#             assert new_node not in self.alias_nodes
#             self.alias_nodes.remove(old_node)
#             self.alias_nodes.add(new_node)

#         def add_node(self, new_node):
#             assert new_node not in self.alias_nodes
#             self.alias_nodes.add(new_node)
            

#         def _infer_buffer_size(self):
#             if type(self.meta_val) == FakeTensor:
#                 self.buffer_size = (
#                     self.meta_val.element_size() * self.meta_val.nelement()
#                 )
#             else:
#                 self.buffer_size = None

#         def _init_first_and_last_alias(self):
#             first_pos = None
#             first_node = None
#             last_pos = None
#             last_node = None

#             for node in self.alias_nodes:
#                 line = self.node2line[node]
                
#                 if first_pos is None or line <= first_pos:
#                     first_node = node
#                     first_pos = line
#                 if last_pos is None or line >= last_pos:
#                     last_node = node
#                     last_pos = line

#             self._first_alias_node = first_node
#             self._last_alias_node = last_node

#         def is_before_and_disjoint(self, val):
#             return self.life_end < val.life_begin

#         def is_same_buffer_size(self, val):
#             if val.buffer_size is None or self.buffer_size is None:
#                 return False
#             return val.buffer_size == self.buffer_size

#         def can_reuse_buffer_of(self, val):
#             assert not val == self
#             assert val.is_before_and_disjoint(self)
#             self._can_reuse_buffer_of.append(val)

#         def try_reuse_buffer(self):
#             if self._reuse_buffer_of is None:
#                 return
#             reuse_node = self._reuse_buffer_of._last_alias_node
#             cur_node = self._first_alias_node
#             print(f"try reuse: {cur_node.op=}, {type(cur_node.target)=}")
#             if (
#                 cur_node.op == "call_function"
#                 and type(cur_node.target) == torch._ops.OpOverload
#                 and cur_node.target.__name__ == "mm.default"
#             ):
#                 # replace to mm.out
#                 with graph.inserting_before(cur_node):
#                     # check dtype
#                     if not get_dtype(reuse_node) == get_dtype(cur_node):
#                         reuse_node = graph.call_function(
#                             torch.ops.aten.view,
#                             (reuse_node, get_dtype(cur_node)),
#                         )
#                     # if not get_shape(reuse_node) == get_shape(cur_node):
#                     reuse_node = graph.call_function(
#                         torch.ops.aten.resize_,
#                         (reuse_node, (0,)),
#                     )
#                     inplace_node = graph.call_function(
#                         torch.ops.aten.mm.out, cur_node.args, {"out": reuse_node}
#                     )
#                     # replace
#                     # TODO: also replace kwargs
#                     users = list(cur_node.users.keys())
#                     for user in users:
#                         replace_args(user, cur_node, inplace_node)
#                     graph.erase_node(cur_node)
#                     print("reuse success!")

#         def contain_one_of(self, nodes):
#             return len(set(nodes) & set(self.alias_nodes)) > 0

#         def show(self):
#             print(f"\nvalue {self.root_node.name}:")
#             alias_names = [node.name for node in self.alias_nodes]
#             alias_names = ",".join(alias_names)
#             print(f"    alias: {alias_names}")
#             print(f"    life_time: [ {self.life_begin} , {self.life_end} ]")
#             print(f"    buffer_size: {self.buffer_size}")

#             can_reuse_buffer_name = [
#                 value.root_node.name for value in self._can_reuse_buffer_of
#             ]
#             can_reuse_buffer_name = ",".join(can_reuse_buffer_name)
#             print(f"    can_reuse_buffer: {can_reuse_buffer_name}")
#             print(f"    reuse_buffer: {self._reuse_buffer_of}")

#     # get node2line
#     node2line = {node: line for line, node in enumerate(graph.nodes)}

#     # get all ssa values
#     alias_union_set = get_alias_union_set(graph)

#     # get each value's life time
#     # root_life_time = {}
#     values = []
#     for root, alias_values in alias_union_set.get_all_alias_values():
#         min_life_begin_time = None
#         max_life_end_time = None
#         for alias in alias_values:
#             life_begin_time, life_end_time = _get_life_time(alias, node2line)
#             min_life_begin_time = (
#                 life_begin_time
#                 if min_life_begin_time is None
#                 else min(min_life_begin_time, life_begin_time)
#             )
#             max_life_end_time = (
#                 life_end_time
#                 if max_life_end_time is None
#                 else max(max_life_end_time, life_end_time)
#             )
#         # assert root not in root_life_time
#         # root_life_time[root] = (min_life_begin_time, max_life_end_time)
#         value = Value(
#             root, alias_values, min_life_begin_time, max_life_end_time, node2line
#         )
#         values.append(value)

#     # find disjoint values
#     # sort by begin time
#     values.sort(key=lambda x: x.life_begin)
#     for i in range(0, len(values)):
#         for j in range(i + 1, len(values)):
#             vi = values[i]
#             vj = values[j]
#             assert not vi == vj
#             assert vi.life_begin <= vj.life_end

#             # judge if vi/vj's life time disjoint
#             disjoint = vi.is_before_and_disjoint(vj)
#             same_buffer_size = vi.is_same_buffer_size(vj)
#             contain_forbid = vi.contain_one_of(forbid_reuse_nodes) or vj.contain_one_of(
#                 forbid_reuse_nodes
#             )
#             if disjoint and same_buffer_size and (not contain_forbid):
#                 vj.can_reuse_buffer_of(vi)
#                 print("can share memory!!")

#     # greedy judge share memorys
#     reused_bufs = set()
#     for value in values:
#         for reuse_buf in value._can_reuse_buffer_of:
#             if reuse_buf not in reused_bufs:
#                 value._reuse_buffer_of = reuse_buf
#                 reused_bufs.add(reuse_buf)
#                 break

#     # share memorys
#     for value in values:
#         value.try_reuse_buffer()

#     # print(graph)
#     # for value in values:
#     #     value.show()
