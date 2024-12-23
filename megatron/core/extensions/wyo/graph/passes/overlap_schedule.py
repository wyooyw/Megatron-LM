import itertools
import torch
from queue import Queue
from ordered_set import OrderedSet
import time
import cProfile
from megatron.core.extensions.wyo.graph.utils import print_rank_0, is_rank_0
# def print_rank_0(text):
#     print(text)
import networkx as nx
import matplotlib.pyplot as plt
from megatron.core.extensions.wyo.model.communicate.communicate import wait_tensor
from megatron.core.extensions.wyo.graph.profiler import BasicProfiler
from torch._subclasses.fake_tensor import FakeTensor
from megatron.core.extensions.wyo.graph.graph_utils import get_fake_tensor
class Node:
    def __init__(self, val):
        self.val = val
        self.users = {}

    def set_users(self, nodes):
        self.users = {key:None for key in nodes}

    def add_user(self, node):
        self.users[node] = None

class ComputationNode(Node):
    def __init__(self, val):
        super().__init__(val)
        self.is_computation_node = True

class CommunicationNode(Node):
    def __init__(self, val):
        super().__init__(val)
        self.is_communication_node = True

class HeavyComputationNode(ComputationNode):
    def __init__(self, val):
        super().__init__(val)
        self.is_heavy_computation_node = True

# class NodeClassifier:
    
#     @staticmethod
#     def is_computation_node(node):
#         return hasattr(node, "is_computation_node") and node.is_computation_node

#     @staticmethod
#     def is_communication_node(node):
#         return hasattr(node, "is_communication_node") and node.is_communication_node

#     @staticmethod
#     def is_heavy_computation_node(node):
#         return hasattr(node, "is_heavy_computation_node") and node.is_heavy_computation_node

class NodeClassifier:

    @staticmethod
    def is_placeholder_node(node):
        return node.op=="placeholder"
    
    @staticmethod
    def is_computation_node(node):
        return not (
            NodeClassifier.is_communication_node(node)
        or  NodeClassifier.is_output(node)
        )
    
    @staticmethod
    def is_communication_node(node):
        if (
            node.op == "call_function"
            and hasattr(node.target, "_opname")
            and node.target._opname in [
                "reduce_scatter_along_first_dim_in_tp_group",
                "gather_along_first_dim_in_tp_group",
                "all_reduce_in_tp_group",
            ]
        ):
            return True
        return False
    
    @staticmethod
    def is_heavy_computation_node(node):
        if (
            node.op == "call_function"
            and hasattr(node.target, "_opname")
            and node.target._opname in [
                "mm", "flash_attn", "flash_attn_bwd", "vocab_parallel_cross_entropy", "vocab_parallel_cross_entropy_backward"
            ]
        ):
            return True
        return False
    
    @staticmethod
    def is_output(node):
        return node.target == "output"
    
    @staticmethod
    def is_super_node(node):
        return NodeClassifier.is_heavy_computation_node(node) or NodeClassifier.is_communication_node(node) or NodeClassifier.is_begin_or_end(node)

    @staticmethod
    def is_begin(node):
        if (
            node.op == "call_function"
            and hasattr(node.target, "_opname")
            and node.target._opname in ["begin"]
        ):
            return True
        return False

    @staticmethod
    def is_end(node):
        if (
            node.op == "call_function"
            and hasattr(node.target, "_opname")
            and node.target._opname in ["end"]
        ):
            return True
        return False
    
    @staticmethod
    def is_begin_or_end(node):
        return NodeClassifier.is_begin(node) or NodeClassifier.is_end(node)

class NodeWrapper:
    def __init__(self, node):
        self.node = node
        self.users = dict()
        self.args = dict()

        name = node.target._opname
        if len(name) > 15:
            name = name[:15]
        self.name = f"{node.__sort_idx__}.{name}"

    def connect_to(self, node):
        self.users[node] = None
        node.args[self] = None

    def connect_from(self, node):
        node.connect_to(self)

    @property
    def op(self):
        return self.node.op
    
    @property
    def target(self):
        return self.node.target

class Graph:
    def __init__(self, nodes):
        self.nodes = OrderedSet(nodes)
        self.graph_key = None

    def num_total_edges(self):
        num = 0
        for node in self.nodes:
            num += len(node.users)
        return num
    
    def show(self, sorted=False):
        nodes = list(self.nodes)

        if sorted:
            nodes.sort(key=lambda x: x.__sort_idx__)

        for node in nodes:
            if NodeClassifier.is_heavy_computation_node(node):
                node_type = "heavy_comp"
            elif NodeClassifier.is_computation_node(node):
                node_type = "comp"
            elif NodeClassifier.is_communication_node(node):
                node_type = "comm"
            else:
                assert False, "Should not happen!"
            # node_val = node.val
            node_val = (str(node.op), str(node.target))
            node_sort_idx = node.__sort_idx__
            print_rank_0(f" {node_type=}, {node_val=}, {node_sort_idx=}")
        

    def topo_sort_and_give_sort_idx(self):
        node_indegree = self.get_node_indegree()
        zero_indegree_nodes = self.get_zero_indegree_nodes()

        sorted_nodes = []

        queue = Queue()
        for node in zero_indegree_nodes:
            queue.put(node)
        while not queue.empty():
            node = queue.get()
            sorted_nodes.append(node)

            for user in node.users.keys():
                if user in node_indegree:
                    node_indegree[user] -= 1
                    assert node_indegree[user]>=0
                    if node_indegree[user]==0:
                        queue.put(user)
        
        for idx, node in enumerate(sorted_nodes):
            node.__sort_idx__ = idx

    def get_graph_with_only_heavy_comp_and_comm(self):

        # step 1: get previous important nodes

        node_indegree = self.get_node_indegree()
        zero_indegree_nodes = self.get_zero_indegree_nodes()

        sorted_nodes = []
        previous_important_nodes = dict()
        
        for node in self.nodes:
            previous_important_nodes[node] = OrderedSet()

        queue = Queue()
        for node in zero_indegree_nodes:
            queue.put(node)
        while not queue.empty():
            node = queue.get()
            # sorted_nodes.append(node)

            if NodeClassifier.is_super_node(node):
                for user in node.users.keys():
                    if user in node_indegree:
                        previous_important_nodes[user].add(node)

                        node_indegree[user] -= 1
                        assert node_indegree[user]>=0
                        if node_indegree[user]==0:
                            queue.put(user)
            elif NodeClassifier.is_computation_node(node):
                for user in node.users.keys():
                    if user in node_indegree:
                        previous_important_nodes[user] |= previous_important_nodes[node]

                        node_indegree[user] -= 1
                        assert node_indegree[user]>=0
                        if node_indegree[user]==0:
                            queue.put(user)

        
        # step 2: construct new graph
        important_nodes_to_wrapper = {node:NodeWrapper(node) for node in self.nodes if (NodeClassifier.is_super_node(node))}
        for node,wrapper in important_nodes_to_wrapper.items():
            for previous_node in previous_important_nodes[node]:
                previous_wrapper = important_nodes_to_wrapper[previous_node]
                wrapper.connect_from(previous_wrapper)

        
        new_nodes = list(important_nodes_to_wrapper.values())
        new_graph = Graph(new_nodes)
        return new_graph
        

    def erase(self, nodes):
        assert type(nodes)==OrderedSet, f"{type(nodes)=}"
        assert nodes.issubset(self.nodes), f"nodes={[node.__sort_idx__ for node in nodes]}, self.nodes={[node.__sort_idx__ for node in self.nodes]}"
        left_nodes = self.nodes - nodes
        return Graph(left_nodes)

    def add_node(self, node):
        assert hasattr(node, "__sort_idx__")
        self.nodes.add(node)

    def not_empty(self):
        return len(self.nodes) > 0
    
    def get_key(self):
        # if self.graph_key is not None:
        #     return self.graph_key
        nodes_list = [node.__sort_idx__ for node in self.nodes]
        nodes_list.sort()
        nodes_tuple = tuple(nodes_list)
        # self.graph_key = nodes_tuple
        return nodes_tuple

    def get_node_indegree(self):
        node_indegree = dict()
        for node in self.nodes:
            node_indegree[node] = 0
        for node in self.nodes:
            for user in node.users.keys():
                if user in node_indegree:
                    node_indegree[user] += 1
        return node_indegree
    
    def get_node_outdegree(self):
        node_outdegree = dict()
        for node in self.nodes:
            node_outdegree[node] = 0
        for node in self.nodes:
            for arg in node.args.keys():
                if arg in node_outdegree:
                    node_outdegree[arg] += 1
        return node_outdegree

    def get_zero_indegree_nodes(self):
        node_indegree = self.get_node_indegree()
        
        zero_indegree_nodes = []
        for node, indegree in node_indegree.items():
            if indegree==0:
                zero_indegree_nodes.append(node)
        return zero_indegree_nodes
    
    def forall_heads(self):
        # step 1: find all nodes with indegree=0
        # node_indegree = self.get_node_indegree()
        zero_indegree_nodes = self.get_zero_indegree_nodes()

        # step 2: judge what stratrgy to use
        #   a. all nodes in zero_indegree_nodes are computation nodes
        #   b. some nodes in zero_indegree_nodes are computation nodes, others are communication nodes
        #   c. all nodes in zero_indegree_nodes are communication nodes
        zero_indegree_comp_nodes = []
        zero_indegree_comm_nodes = []
        for node in zero_indegree_nodes:
            if NodeClassifier.is_computation_node(node):
                zero_indegree_comp_nodes.append(node)
            elif NodeClassifier.is_communication_node(node):
                zero_indegree_comm_nodes.append(node)
            else:
                assert False, f"This should not happen. {node.op=}, {node.target=}"
        # print(f"{len(zero_indegree_comp_nodes)=}")
        # print(f"{len(zero_indegree_comm_nodes)=}")
        # print_rank_0("zero_indegree_comp_nodes:")
        # for node in zero_indegree_comp_nodes:
        #     print_rank_0(f"  {node.op=}, {node.target=}")
        # print_rank_0("zero_indegree_comm_nodes:")
        # for node in zero_indegree_comm_nodes:
        #     print_rank_0(f"  {node.op=}, {node.target=}")

        if len(zero_indegree_comp_nodes) > 0 and len(zero_indegree_comm_nodes) > 0:
            return self._enumerate_compution_communication_overlap_head(zero_indegree_comp_nodes, zero_indegree_comm_nodes)
        elif len(zero_indegree_comp_nodes) > 0 and len(zero_indegree_comm_nodes) == 0:
            return self._enumerate_pure_compution_head(zero_indegree_comp_nodes)
        elif len(zero_indegree_comp_nodes) == 0 and len(zero_indegree_comm_nodes) > 0:
            return self._enumerate_pure_communication_head(zero_indegree_comm_nodes)
        else:
            return []

    def _enumerate_pure_communication_head(self, zero_indegree_comm_nodes):
        for node in zero_indegree_comm_nodes:
            yield PureCommunicationSubGraph(node)

    def _enumerate_pure_compution_head(self, zero_indegree_comp_nodes):
        # print("_enumerate_pure_compution_head")
        """
        enumerate along each zero-indegree computation nodes, until meet:
            1) heavy computation node, such as matmul and attention, or
            2) branch, or
            3) communication node, or
            4) the end of graph, or
            5) next node has indegree > 1
        """
        node_indegree = self.get_node_indegree()
        for node in zero_indegree_comp_nodes:
            head_list = self._enumerate_pure_computation_head_begin_with_node(node, node_indegree)
            for head in head_list:
                yield PureComputionSubGraph(head)

    def _enumerate_pure_computation_head_begin_with_node(self, begin_node, node_indegree, must_contain_heavy_comp_node=False):
        # print("_enumerate_pure_computation_head_begin_with_node")
        head_list = []
        head = []
        max_heavy_node = 2
        heavy_node = 0
        cur_node = begin_node
        # contain_heavy_node = False
        while True:
            head.append(cur_node)

            if NodeClassifier.is_heavy_computation_node(cur_node):
                heavy_node += 1
                head_list.append(head.copy())

            if (
                # NodeClassifier.is_heavy_computation_node(cur_node)
                heavy_node >= max_heavy_node
                or len(cur_node.users) > 1
                or len(cur_node.users) == 0
            ):
                break
            
            assert len(cur_node.users)==1
            cur_node = list(cur_node.users.keys())[0]
            
            if (
                cur_node not in node_indegree
                or node_indegree[cur_node] > 1
                or NodeClassifier.is_communication_node(cur_node)
            ):
                break

        if heavy_node >= 0 and must_contain_heavy_comp_node:
            return None

        if not (len(head_list) > 0 and len(head)==len(head_list[-1])):
            head_list.append(head)
        
        return head_list

    def _enumerate_compution_communication_overlap_head(self, zero_indegree_comp_nodes, zero_indegree_comm_nodes):
        
        node_indegree = self.get_node_indegree()
        
        # TODO: purn this space
        comp_heads = []
        for comp_node in zero_indegree_comp_nodes:
            comp_head_list = self._enumerate_pure_computation_head_begin_with_node(
                begin_node=comp_node,
                node_indegree=node_indegree
            )
            comp_heads.extend(comp_head_list)
        # print(f"{len(comp_heads)=}")
        # pick one comm_node
        for comm_node in zero_indegree_comm_nodes:
            
            # try all subset of comp_nodes
            for subset_len in range(1, len(comp_heads) + 1):
            # for subset_len in range(1, min(len(comp_heads) + 1, 2)):
                for comp_heads_subset in itertools.combinations(comp_heads, subset_len):
                    comp_nodes = [item for sublist in comp_heads_subset for item in sublist]
                    yield ComputationCommunicationOverlapSubGraph(comm_node=comm_node, comp_nodes=comp_nodes)

class ComputationCommunicationOverlapSubGraph:
    def __init__(self, comm_node, comp_nodes):
        assert comp_nodes is None or type(comp_nodes)==list
        self.comm_node = comm_node
        self.comp_nodes = comp_nodes

    def _show(self):
        self._show_detail()
        return
        if self.comp_nodes is not None:
            # comp_nodes_val = [f"({node.val})" for node in self.comp_nodes]
            comp_nodes_val = [f"({node.name})" for node in self.comp_nodes]
            comp_nodes_idx = [node.__sort_idx__ for node in self.comp_nodes]
            print_rank_0(f"    comp nodes: val={comp_nodes_val}, idx={comp_nodes_idx}")
        
        if self.comm_node is not None:
            # print_rank_0(f"    comm nodes: {self.comm_node.val}, idx={self.comm_node.__sort_idx__}")
            print_rank_0(f"    comm nodes: {self.comm_node.name}, idx={self.comm_node.__sort_idx__}")

    def _show_detail(self):
        if self.comp_nodes is not None:
            print_rank_0(f"  Comp Nodes")
            for comp_node in self.comp_nodes:
                print_rank_0(f"    - name: {comp_node.name},")
                print_rank_0(f"    - users:")
                for user in comp_node.users.keys():
                    print_rank_0(f"        {user.name}")
        
        if self.comm_node is not None:
            print_rank_0(f"  Comm Nodes")
            print_rank_0(f"    - name: {self.comm_node.name},")
            print_rank_0(f"    - users:")
            for user in self.comm_node.users.keys():
                print_rank_0(f"      - {user.name}")

    def show(self):
        print_rank_0("Comm Comp Overlap Head:")
        self._show()

    def score(self):
        comm_score = 0
        if self.comm_node is not None:
            comm_score = 1

        comp_score = 0
        if self.comp_nodes is not None:
            for node in self.comp_nodes:
                if NodeClassifier.is_heavy_computation_node(node):
                    comp_score += 1
                elif NodeClassifier.is_computation_node(node):
                    comp_score += 0.01
        return max(comm_score, comp_score)

    def get_nodes(self):
        nodes = OrderedSet()
        if self.comm_node is not None:
            nodes.add(self.comm_node)
        
        if self.comp_nodes is not None:
            for node in self.comp_nodes:
                nodes.add(node)
        
        return nodes
        
        

class PureComputionSubGraph(ComputationCommunicationOverlapSubGraph):
    def __init__(self, comp_nodes):
        super().__init__(None, comp_nodes)
    
    @staticmethod
    def merge(self, pure_comp_subgraphs):
        all_comp_nodes = []
        for pure_comp_subgraph in pure_comp_subgraphs:
            all_comp_nodes.extend(pure_comp_subgraph.comp_nodes)
        return PureComputionSubGraph(all_comp_nodes)

    def show(self):
        print_rank_0("Comp Head:")
        self._show()
            
            
class PureCommunicationSubGraph(ComputationCommunicationOverlapSubGraph):
    def __init__(self, comm_node):
        super().__init__(comm_node, None)

    def show(self):
        print_rank_0("Comm Head:")
        self._show()



def get_output_node(graph):
    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break
    return output_node

def get_fake_tensor_of(node):
    if not isinstance(node, torch.fx.graph.Node):
        return None
    fake_tensor = node.meta.get(
        "val",
        node.meta.get(
            "tensor_meta", node.meta.get("example_value", None)
        ),
    )
    if isinstance(fake_tensor, FakeTensor):
        return fake_tensor
    else:
        return None

class OverlapScheduler:
    def __init__(self):
        self.record_score = dict()
        self.record_head = dict()
        self.profiler = BasicProfiler()
        self.profiler.begin_basic_profile()
        
    
    def record_result(self, graph, head, score):
        graph_key = graph.get_key()
        self.record_score[graph_key] = score
        self.record_head[graph_key] = head

    def get_node_score(self, node):
        # assert isinstance(node, NodeWrapper), f"{type(node)=}"
        # print_rank_0(f"{node.name=}")
        if isinstance(node, NodeWrapper):
            node = node.node

        comm_name = node.target._opname
        comm_inputs = [get_fake_tensor_of(arg) for arg in node.args]
        comm_inputs = tuple(comm_inputs)
        comm_time = self.profiler.get_time(comm_name, comm_inputs)
        # print_rank_0(f"get_node_score={comm_time=}")
        return comm_time


    def get_head_score(self, head):
        # return head.score()
        if isinstance(head, PureComputionSubGraph):
            comp_time = 0
            for comp_node in head.comp_nodes:
                comp_time += self.get_node_score(comp_node)
            return comp_time, {"comp": comp_time, "comm": 0}
        elif isinstance(head, PureCommunicationSubGraph):
            comm_time = self.get_node_score(head.comm_node)
            return comm_time, {"comp": 0, "comm": comm_time}
        elif isinstance(head, ComputationCommunicationOverlapSubGraph):
            assert head.comm_node is not None
            comm_time = self.get_node_score(head.comm_node)
            comp_time = 0
            for comp_node in head.comp_nodes:
                comp_time += self.get_node_score(comp_node)
            return max(comm_time, comp_time), {"comp": comp_time, "comm": comm_time}
        else:
            assert False, "This should not happen!"

    def find_best_partition_strategy(self, graph):
        
        # print("---------------------------------")
        # print("find_best_partition_strategy")
        # graph.show()
        # print("---------------------------------")
        if not graph.not_empty():
            return 0
        
        graph_key = graph.get_key()
        if graph_key in self.record_score:
            return self.record_score[graph_key]
        self._search_cnt += 1
        
        if self._search_cnt % 10000 == 0:
            print_rank_0(f"{self._search_cnt=}")

        min_score = 9999999999
        min_head = None
        for head in graph.forall_heads():
            head_score,_ = self.get_head_score(head)
            left_graph = graph.erase(head.get_nodes())
            left_graph_score = self.find_best_partition_strategy(left_graph)
            # print(f"{head_score=}, {left_graph_score=}")
            if head_score + left_graph_score < min_score:
                min_score = head_score + left_graph_score
                min_head = head
        # print_rank_0(f"find best partition: {min_score=}")
        # assert min_head is not None

        self.record_result(graph, min_head, min_score)
        return min_score

    def get_partition_from_record_head(self, graph):
        
        head_list = []

        while graph.not_empty():
            graph_key = graph.get_key()
            head = self.record_head[graph_key]
            head_list.append(head)
            graph = graph.erase(head.get_nodes())
            
        
        return head_list
    
    def schedule_overlap(self, graph):
        self._search_cnt = 0
        # 
        print("schedule_overlap begin")
        self.find_best_partition_strategy(graph)
        # exit()
        print("find_best_partition_strategy finish")
        # print("")
        # print(f"{self.record_score=}")
        # print("")
        # print(f"{self.record_head=}")
        # exit()
        torch.distributed.barrier()
        head_list = self.get_partition_from_record_head(graph)
        print_rank_0(f"Show heads begin")
        for head in head_list:
            head.show()
            score, detail = self.get_head_score(head)
            print_rank_0(f"    {score=}, {detail=}")
        print_rank_0(f"Show heads end")
        torch.distributed.barrier()
        # exit()
        # print("get_partition_from_record_head finish")
        return head_list
        fused_graph = torch.fx.graph.Graph()
        value_remap = {}

        def _add_node_to(node, graph, value_remap):
            assert not NodeClassifier.is_output(node), "Node is output, should not pass in this function!"
            value_remap[node] = graph.node_copy(node, lambda n: value_remap[n])
            if value_remap[node].op == "placeholder":
                value_remap[node].target = value_remap[node].name

        def _add_nodes_to(nodes, graph, value_remap):
            nodes = list(nodes)
            nodes.sort(key=lambda x: x.__sort_idx__)
            for node in nodes :
                _add_node_to(node, graph, value_remap)

        def _add_async_comm_node_to(node, graph, value_remap):
            args = [value_remap[arg] for arg in node.args]
            async_comm_node = graph.call_function(node.target, args=tuple(args), kwargs={"async_op": True})
            wait_node = graph.call_function(wait_tensor, args=(async_comm_node,))
            value_remap[node] = wait_node
            return async_comm_node, wait_node
                
        for head in head_list:
            if isinstance(head, PureComputionSubGraph):
                _add_nodes_to(head.comp_nodes, fused_graph, value_remap)
            elif isinstance(head, PureCommunicationSubGraph):
                _add_node_to(head.comm_node, fused_graph, value_remap)
            elif isinstance(head, ComputationCommunicationOverlapSubGraph):
                # convert communication node into async version
                async_comm_node, wait_node = _add_async_comm_node_to(head.comm_node, fused_graph, value_remap)

                # add all comp nodes
                with fused_graph.inserting_after(async_comm_node):
                    _add_nodes_to(head.comp_nodes, fused_graph, value_remap)

            else:
                assert False, "This should not happen!"

        # build output node
        fused_graph.output(get_output_node(graph).args[0])
        
        return fused_graph

def draw_new_graph(node_wrappers):
    if is_rank_0():
        # 创建一个有向图
        G = nx.DiGraph()

        for wrapper in node_wrappers:
            G.add_node(wrapper.name)

        for wrapper in node_wrappers:
            for user_wrapper in wrapper.users:
                G.add_edge(wrapper.name, user_wrapper.name)
        
        pos = nx.spring_layout(G, k=2, iterations=100)  # 为图形设置布局
        nx.draw(G, pos, with_labels=True, node_size=150, font_size=8) 

        plt.savefig("graph.jpeg")

def overlap_schedule(graph):
    # node.op == "call_function"
    #         and hasattr(node.target, "_opname")
    #         and node.target._opname 

    # for node in graph.nodes:
    #     if node.op=="call_function" and hasattr(node.target, "_opname"):
    #         print_rank_0(f"{node.target._opname=}")
    # exit()
    # turn fx graph into graph
    nodes = []
    placeholder_nodes = []
    output_nodes = []
    for node in graph.nodes:
        if not (NodeClassifier.is_output(node) or NodeClassifier.is_placeholder_node(node)):
            nodes.append(node)
        elif NodeClassifier.is_placeholder_node(node):
            placeholder_nodes.append(node)
        elif NodeClassifier.is_output(node):
            output_nodes.append(node)
        else:
            assert False, "This should not happen!"
        
    graph = Graph(nodes)
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)

    print_rank_0("------------------------------------")

    super_graph = graph.get_graph_with_only_heavy_comp_and_comm()
    super_graph.topo_sort_and_give_sort_idx()
    super_graph.show(sorted=True)
    print_rank_0(f"before {super_graph.num_total_edges()=}")
    add_virtual_edges_in_super_graph(super_graph)
    print_rank_0(f"after {super_graph.num_total_edges()=}")
    # exit()
    # fused_graph = generate_fx_graph_from_two_graphs(placeholder_nodes, output_nodes, graph, super_graph)
    # return fused_graph
    # print_rank_0(f"{len(new_graph)=}")
    # for node_wrapper in new_graph:
    #     node = node_wrapper.node
    #     print_rank_0(f"{node.target._opname=} {len(node_wrapper.users)=} {len(node_wrapper.args)=}")

    # draw_new_graph(new_graph)
    

    # print_rank_0("\n----------- get_node_indegree ----------")
    # node_indegree = graph.get_node_indegree()
    # for node, indegree in node_indegree.items():
    #     print_rank_0(f"{node.op=}, {node.target=}, {indegree=}")
    # exit()

    # for head in graph.forall_heads():
    #     head.show()

    scheduler = OverlapScheduler()
    begin_time = time.time()
    super_graph_heads = scheduler.schedule_overlap(super_graph)
    end_time = time.time()
    
    search_time = end_time - begin_time
    print(f"{scheduler._search_cnt=}")
    print(f"{search_time=} s")

    fused_graph = generate_fx_graph_from_super_graph_heads(
        placeholder_nodes,
        output_nodes,
        graph,
        super_graph,
        super_graph_heads
    )

    # fused_graph = generate_fx_graph_from_two_graphs(placeholder_nodes, output_nodes, graph, fused_super_graph)
    # exit()
    return fused_graph


def generate_fx_graph_from_two_graphs(placeholder_nodes, output_nodes, graph, super_graph):
    node_indegree = graph.get_node_indegree()

    super_nodes = list(super_graph.nodes)
    super_nodes.sort(key=lambda x: x.__sort_idx__)
    fx_nodes = []

    def _iter_common_node_from_begin_nodes(begin_nodes, node_indegree):
        result_nodes = []
        queue = Queue()
        for node in begin_nodes:
            queue.put(node)
        while not queue.empty():
            node = queue.get()
            if NodeClassifier.is_super_node(node):
                continue
            result_nodes.append(node)

            for user in node.users.keys():
                if user in node_indegree:
                    node_indegree[user] -= 1
                    assert node_indegree[user] >= 0
                    if node_indegree[user]==0:
                        queue.put(user)

        return result_nodes

    # add all common nodes before super_nodes
    zero_indegree_nodes = graph.get_zero_indegree_nodes()
    fx_nodes.extend(_iter_common_node_from_begin_nodes(zero_indegree_nodes, node_indegree))
    # print_rank_0("----------------------")
    # print_rank_0(f"{len(fx_nodes)=}")
    for super_node in super_nodes:
        # add super_node into fx graph
        node = super_node.node
        # print_rank_0(f"{super_node.name=},   {node_indegree[node]=}")
        assert node_indegree[node]==0, f"{node_indegree[node]=}"
        fx_nodes.append(node)

        begin_nodes = []
        for user in node.users.keys():
            if user in node_indegree:
                node_indegree[user] -= 1
                assert node_indegree[user] >= 0
                if node_indegree[user]==0:
                    begin_nodes.append(user)
        fx_nodes.extend(_iter_common_node_from_begin_nodes(begin_nodes, node_indegree))

    assert len(fx_nodes)==len(graph.nodes)
    print_rank_0(f"{len(fx_nodes)=}")
    # 2. construct fx graph
    fused_graph = torch.fx.graph.Graph()
    value_remap = {}

    def _add_node_to(node, graph, value_remap):
        assert not NodeClassifier.is_output(node), "Node is output, should not pass in this function!"
        value_remap[node] = graph.node_copy(node, lambda n: value_remap[n])
        if value_remap[node].op == "placeholder":
            value_remap[node].target = value_remap[node].name

    # 2.1 add placeholders
    for node in placeholder_nodes:
        _add_node_to(node, fused_graph, value_remap)

    # 2.2 add computation/communication ops
    for node in fx_nodes:
        _add_node_to(node, fused_graph, value_remap)

    # 2.3 add output ops
    assert len(output_nodes)==1
    output_args = output_nodes[0].args[0]
    remap_output_args = []
    for arg in output_args:
        remap_output_args.append(None if arg is None else value_remap[arg])
    fused_graph.output(remap_output_args)

    return fused_graph

def generate_fx_graph_from_super_graph_heads(
        placeholder_nodes, 
        output_nodes, 
        graph, 
        super_graph,
        super_graph_heads,
        ):
    node_indegree = graph.get_node_indegree()
    super_graph.topo_sort_and_give_sort_idx()
    
    fx_nodes = []
    def _iter_common_node_from_begin_nodes(begin_nodes, node_indegree):
        result_nodes = []
        queue = Queue()
        for node in begin_nodes:
            queue.put(node)
        while not queue.empty():
            node = queue.get()
            if NodeClassifier.is_super_node(node):
                continue
            result_nodes.append(node)

            for user in node.users.keys():
                if user in node_indegree:
                    node_indegree[user] -= 1
                    assert node_indegree[user] >= 0
                    if node_indegree[user]==0:
                        queue.put(user)

        return result_nodes

    def _add_node_to(node, graph, value_remap):
        assert not NodeClassifier.is_output(node), "Node is output, should not pass in this function!"
        value_remap[node] = graph.node_copy(node, lambda n: value_remap[n])
        if value_remap[node].op == "placeholder":
            value_remap[node].target = value_remap[node].name

    def _add_nodes_to(nodes, graph, value_remap):
        nodes = list(nodes)
        nodes.sort(key=lambda x: x.__sort_idx__)
        for node in nodes :
            _add_node_to(node, graph, value_remap)

    def _add_async_comm_node_to(node, graph, value_remap):
        args = [value_remap[arg] for arg in node.args]
        async_comm_node = graph.call_function(node.target, args=tuple(args), kwargs={"async_op": True})
        wait_node = graph.call_function(wait_tensor, args=(async_comm_node,))

        async_comm_node.meta["val"] = get_fake_tensor(node).clone()
        wait_node.meta["val"] = get_fake_tensor(node).clone()

        value_remap[node] = wait_node
        return async_comm_node, wait_node

    def _iter_users_of(node, node_indegree):
        begin_nodes = []
        for user in node.users.keys():
            if user in node_indegree:
                node_indegree[user] -= 1
                assert node_indegree[user] >= 0
                if node_indegree[user]==0:
                    begin_nodes.append(user)
        return begin_nodes
    
    fused_graph = torch.fx.graph.Graph()
    value_remap = {}

    # step 1. add placeholder
    for node in placeholder_nodes:
        _add_node_to(node, fused_graph, value_remap)

    # step 2: add computation / communication nodes
    # add all common nodes before super_nodes
    zero_indegree_nodes = graph.get_zero_indegree_nodes()
    _add_nodes_to(_iter_common_node_from_begin_nodes(zero_indegree_nodes, node_indegree), fused_graph, value_remap)
    print_rank_0("Begin build body!")
    for head in super_graph_heads:
        # head.show()
        if isinstance(head, PureComputionSubGraph):

            super_nodes = list(head.comp_nodes)
            super_nodes.sort(key=lambda x: x.__sort_idx__)
            for super_node in super_nodes :
                node = super_node.node
                _add_node_to(node, fused_graph, value_remap)
                begin_nodes = _iter_users_of(node, node_indegree)
                _add_nodes_to(_iter_common_node_from_begin_nodes(begin_nodes, node_indegree), fused_graph, value_remap)

        elif isinstance(head, PureCommunicationSubGraph):
            
            super_node = head.comm_node
            node = super_node.node
            _add_node_to(node, fused_graph, value_remap)
            begin_nodes = _iter_users_of(node, node_indegree)
            _add_nodes_to(_iter_common_node_from_begin_nodes(begin_nodes, node_indegree), fused_graph, value_remap)

        elif isinstance(head, ComputationCommunicationOverlapSubGraph):
            # convert communication node into async version
            async_comm_node, wait_node = _add_async_comm_node_to(head.comm_node.node, fused_graph, value_remap)

            # add all comp nodes
            with fused_graph.inserting_before(wait_node):
                super_nodes = list(head.comp_nodes)
                super_nodes.sort(key=lambda x: x.__sort_idx__)
                for super_node in super_nodes :
                    node = super_node.node
                    _add_node_to(node, fused_graph, value_remap)
                    begin_nodes = _iter_users_of(node, node_indegree)
                    _add_nodes_to(_iter_common_node_from_begin_nodes(begin_nodes, node_indegree), fused_graph, value_remap)
                
            begin_nodes = _iter_users_of(head.comm_node.node, node_indegree)
            _add_nodes_to(_iter_common_node_from_begin_nodes(begin_nodes, node_indegree), fused_graph, value_remap)

        else:
            assert False, "This should not happen!"

    # 2.3 add output ops
    assert len(output_nodes)==1
    output_args = output_nodes[0].args[0]
    remap_output_args = []
    for arg in output_args:
        remap_output_args.append(None if arg is None else value_remap[arg])
    fused_graph.output(remap_output_args)

    return fused_graph

def _get_node_distance_from_begin(begin_node, super_graph):
    """
    Current version: shortest path
    TODO: Use longest path is better.
    """

    # get a begin-node-reachable subgraph from super_graph
    begin_reachable_nodes = OrderedSet()
    queue = Queue()
    queue.put(begin_node)
    while not queue.empty():
        super_node = queue.get()
        begin_reachable_nodes.add(super_node)

        for arg in super_node.users.keys():
            if arg not in begin_reachable_nodes:
                queue.put(arg)

    print_rank_0(f"{len(begin_reachable_nodes)=}")
    super_graph = Graph(begin_reachable_nodes)


    queue = Queue()
    queue.put(begin_node)
    
    node_indegree = super_graph.get_node_indegree()
    # print_rank_0("\n------------ node_indegree ------------")
    # for node, indegree in node_indegree.items():
    #     print_rank_0(f"{node.name}, {indegree=}")
    # print_rank_0("------------ ------------ ------------\n")

    # print_rank_0("\n------------ begin_node.users ------------")
    # for user in begin_node.users.keys():
    #     print_rank_0(f"{user.name}")
    # print_rank_0("------------ ------------ ------------\n")

    # print_rank_0("\n------------ begin_node.args ------------")
    # for arg in begin_node.args.keys():
    #     print_rank_0(f"{arg.name}")
    # print_rank_0("------------ ------------ ------------\n")

    node_distance_from_begin = {begin_node: 0}
    road = {begin_node: None}
    while not queue.empty():
        super_node = queue.get()
        for user in super_node.users.keys():
            if user not in node_indegree:
                continue
            node_indegree[user] -= 1
            if user in node_distance_from_begin:
                node_distance_from_begin[user] = max(node_distance_from_begin[user], node_distance_from_begin[super_node]+1)
            else:
                node_distance_from_begin[user] = node_distance_from_begin[super_node]+1

            assert node_indegree[user] >= 0, f"{super_node.name=},  {user.name=},  {node_indegree[user]=}"
            if node_indegree[user]==0:
                queue.put(user)
                road[user] = super_node

    return node_distance_from_begin, road

# def _get_node_can_reach_end(end_node, super_graph):
#     queue = Queue()
#     queue.put(end_node)
    
#     nodes_can_reach_end = set()
#     while not queue.empty():
#         super_node = queue.get()
#         nodes_can_reach_end.add(super_node)

#         for arg in super_node.args.keys():
#             if arg not in nodes_can_reach_end:
#                 queue.put(arg)

#     return nodes_can_reach_end

def _get_node_distance_to_end(end_node, super_graph):
    """
    Current version: shortest path
    TODO: Use longest path is better.
    """

    # get a begin-node-reachable subgraph from super_graph
    end_reachable_nodes = OrderedSet()
    queue = Queue()
    queue.put(end_node)
    while not queue.empty():
        super_node = queue.get()
        end_reachable_nodes.add(super_node)

        for arg in super_node.args.keys():
            if arg not in end_reachable_nodes:
                queue.put(arg)
    super_graph = Graph(end_reachable_nodes)


    queue = Queue()
    queue.put(end_node)
    
    node_outdegree = super_graph.get_node_outdegree()
    node_distance_to_end = {end_node: 0}
    road = {end_node: None}
    while not queue.empty():
        super_node = queue.get()
        for user in super_node.args.keys():
            if user not in node_outdegree:
                continue

            node_outdegree[user] -= 1
            if user in node_distance_to_end:
                node_distance_to_end[user] = max(node_distance_to_end[user], node_distance_to_end[super_node]+1)
            else:
                node_distance_to_end[user] = node_distance_to_end[super_node]+1

            assert node_outdegree[user] >= 0
            if node_outdegree[user]==0:
                queue.put(user)
                road[user] = super_node

    return node_distance_to_end, road

def add_virtual_edges_in_super_graph(super_graph):
    # find first and end nodes
    super_graph.show()
    begin_nodes = []
    end_nodes = []
    for super_node in super_graph.nodes:
        if NodeClassifier.is_begin(super_node.node):
            begin_nodes.append(super_node)
        elif NodeClassifier.is_end(super_node.node):
            end_nodes.append(super_node)
    assert len(begin_nodes)==2, f"{len(begin_nodes)=}"
    assert len(end_nodes)==2, f"{len(end_nodes)=}"

    # print_rank_0("user of begin_nodes[0]:")
    # for user in begin_nodes[0].users.keys():
    #     print_rank_0(f"  {user.name=}")
    #     for arg in user.args.keys():
    #         print_rank_0(f"    {arg.name=}")

    # print_rank_0("user of begin_nodes[1]:")
    # for user in begin_nodes[1].users.keys():
    #     print_rank_0(f"  {user.name=}")
    #     for arg in user.args.keys():
    #         print_rank_0(f"    {arg.name=}")

    # exit()
    # remove nodes that before begin_nodes and after end_nodes
    # these nodes are not consider
    node_distance_to_begin_0, _ = _get_node_distance_to_end(begin_nodes[0], super_graph)
    node_distance_to_begin_1, _ = _get_node_distance_to_end(begin_nodes[1], super_graph)
    node_distance_from_end_0, _ = _get_node_distance_from_begin(end_nodes[0], super_graph)
    node_distance_from_end_1, _ = _get_node_distance_from_begin(end_nodes[1], super_graph)
    erase_nodes = OrderedSet()
    for node in [*list(node_distance_to_begin_0.keys()), *list(node_distance_to_begin_1.keys()), *list(node_distance_from_end_0.keys()), *list(node_distance_from_end_1.keys())]:
        if node not in begin_nodes and node not in end_nodes:
            erase_nodes.add(node)
    super_graph = super_graph.erase(erase_nodes)

    # get distances from begins to nodes
    node_distance_from_begin_0, road_from_begin_0 = _get_node_distance_from_begin(begin_nodes[0], super_graph)
    node_distance_from_begin_1, road_from_begin_1 = _get_node_distance_from_begin(begin_nodes[1], super_graph)

    # get nodes that can reach ends
    # nodes_can_reach_end_0 = _get_node_can_reach_end(end_nodes[0], super_graph)
    # nodes_can_reach_end_1 = _get_node_can_reach_end(end_nodes[1], super_graph)
    node_distance_to_end_0, road_to_end_0 = _get_node_distance_to_end(end_nodes[0], super_graph)
    node_distance_to_end_1, road_to_end_1 = _get_node_distance_to_end(end_nodes[1], super_graph)

    # get relation between begins and ends
    if end_nodes[0] in node_distance_from_begin_0:
        assert end_nodes[0] not in node_distance_from_begin_1
        assert end_nodes[1] in node_distance_from_begin_1
        assert end_nodes[1] not in node_distance_from_begin_0
    elif end_nodes[0] in node_distance_from_begin_1:
        assert end_nodes[0] not in node_distance_from_begin_0
        assert end_nodes[1] in node_distance_from_begin_0
        assert end_nodes[1] not in node_distance_from_begin_1
        t = end_nodes[0]
        end_nodes[0] = end_nodes[1]
        end_nodes[1] = t

    # build main roads
    # begin_nodes[0], ..., end_nodes[0]
    # begin_nodes[1], ..., end_nodes[1]
    def build_main_road(begin_node, end_node, road):
        main_road = []
        cur_node = end_node
        while cur_node is not None:
            main_road.append(cur_node)
            assert cur_node in road
            cur_node = road[cur_node]
        assert main_road[-1]==begin_node

        # reverse, from begin to end
        main_road.reverse()
        assert main_road[0]==begin_node

        return main_road

    main_road_0 = build_main_road(begin_nodes[0], end_nodes[0], road_from_begin_0)
    main_road_1 = build_main_road(begin_nodes[1], end_nodes[1], road_from_begin_1)
    print_rank_0(f"{len(main_road_0)=}")
    print_rank_0(f"{len(main_road_1)=}")

    categories = dict()
    for i in range(10):
        new_categories = _add_virtual_edges_in_super_graph(
            super_graph,
            begin_nodes, 
            end_nodes,
            main_road_0,
            main_road_1,
            categories
        )
        print_rank_0(f"{len(new_categories)=}")
        if len(new_categories)==0:
            break
        categories.update(new_categories)
    
    assert len(categories)==len(super_graph.nodes)
    print_rank_0("categories:")
    for node, category in categories.items():
        print_rank_0(f"    {node.name=}, {category=}")
        assert category in (0, 6)

    _add_virtual_edges_for_even_overlap(main_road_0, main_road_1)

    
    

def _add_virtual_edges_in_super_graph(
        super_graph,
        begin_nodes, 
        end_nodes,
        main_road_0,
        main_road_1,
        categories
    ):
    # print_rank_0(f"    {len(begin_nodes[0].args)=}")
    print_rank_0(f"    {len(begin_nodes[1].args)=}")
    nodes_of_main_road_0 = set(main_road_0)
    nodes_of_main_road_1 = set(main_road_1)
    nodes_of_main_road = nodes_of_main_road_0 | nodes_of_main_road_1

    # get distances from begins to nodes
    node_distance_from_begin_0, road_from_begin_0 = _get_node_distance_from_begin(begin_nodes[0], super_graph)
    node_distance_from_begin_1, road_from_begin_1 = _get_node_distance_from_begin(begin_nodes[1], super_graph)

    # get nodes that can reach ends
    node_distance_to_end_0, road_to_end_0 = _get_node_distance_to_end(end_nodes[0], super_graph)
    node_distance_to_end_1, road_to_end_1 = _get_node_distance_to_end(end_nodes[1], super_graph)

    # for each node, get its category
    #   0. is main
    #   1. from main
    #   2. into main
    # do this until no change. Then find
    #   3. independent from main
    IS_MAIN = 0
    FROM_MAIN_0 = 1
    TO_MAIN_0 = 2
    FROM_MAIN_1 = 3
    TO_MAIN_1 = 4
    INDEPENDENT_FROM_MAIN = 5
    OTHER = 6

    new_categories = dict()
    for super_node in super_graph.nodes:
        old_category = categories.get(super_node, None)
        if super_node in nodes_of_main_road:
            new_category = IS_MAIN
        elif super_node in node_distance_from_begin_0 and super_node not in node_distance_to_end_0:
            new_category = FROM_MAIN_0
        elif super_node not in node_distance_from_begin_0 and super_node in node_distance_to_end_0:
            new_category = TO_MAIN_0
        elif super_node in node_distance_from_begin_1 and super_node not in node_distance_to_end_1:
            new_category = FROM_MAIN_1
        elif super_node not in node_distance_from_begin_1 and super_node in node_distance_to_end_1:
            new_category = TO_MAIN_1
        elif (
            super_node not in node_distance_from_begin_0
            and super_node not in node_distance_to_end_0
            and super_node not in node_distance_from_begin_1
            and super_node not in node_distance_to_end_1
        ):
            new_category = INDEPENDENT_FROM_MAIN
        else:
            new_category = OTHER
            
        if not old_category==new_category:
            new_categories[super_node] = new_category
            has_change = True
    
    # add virtual edgs for nodes that change categories
    # print_rank_0("main_road_0:")
    # for node in main_road_0:
    #     print_rank_0(f"    {node.name}")
    # print_rank_0("road_from_begin_0:")
    # for dst,src in road_from_begin_0.items():
    #     src_name = src.name if src is not None else None
    #     print_rank_0(f"    {src_name} -> {dst.name}")
    # print_rank_0("node_distance_from_begin_0")
    # for node,dis in node_distance_from_begin_0.items():
    #     print_rank_0(f"    {node.name} : {dis}")

    for node, category in new_categories.items():
        print_rank_0(f"{node.name=}  {category=}")
        
        if category==FROM_MAIN_0:
            to_main_node = _get_to_main_node(node, nodes_of_main_road_0, main_road_0, road_from_begin_0)
            node.connect_to(to_main_node)
            # print_rank_0(f"    category=FROM_MAIN_0  {node.name} -> {to_main_node.name}")
        elif category==TO_MAIN_0:
            from_main_node = _get_from_main_node(node, nodes_of_main_road_0, main_road_0, road_to_end_0)
            from_main_node.connect_to(node)
            # print_rank_0(f"    category=TO_MAIN_0  {from_main_node.name} -> {node.name}")
        elif category==FROM_MAIN_1:
            to_main_node = _get_to_main_node(node, nodes_of_main_road_1, main_road_1, road_from_begin_1)
            node.connect_to(to_main_node)
            # print_rank_0(f"    category=FROM_MAIN_1  {node.name} -> {to_main_node.name}")
        elif category==TO_MAIN_1:
            from_main_node = _get_from_main_node(node, nodes_of_main_road_1, main_road_1, road_to_end_1)
            # if from_main_node.name=="83.begin":
            #     print_rank_0(f"    {len(from_main_node.args)=}")
            from_main_node.connect_to(node)
            # print_rank_0(f"    category=TO_MAIN_1  {from_main_node.name} -> {node.name}")
            # if from_main_node.name=="83.begin":
            #     print_rank_0(f"    {len(from_main_node.args)=}")

    return new_categories

def _add_virtual_edges_for_even_overlap(
    main_road_0,
    main_road_1
):
    num_edges = 20
    size0 = len(main_road_0)
    size1 = len(main_road_1)

    subsize0 = size0 // num_edges
    subsize1 = size1 // num_edges

    connect_nodes_0 = []
    for i in range(0, size0-subsize0, subsize0):
        connect_nodes_0.append((main_road_0[i], main_road_0[i+1]))

    connect_nodes_1 = []
    for i in range(0, size1-subsize1, subsize1):
        connect_nodes_1.append((main_road_1[i], main_road_1[i+1]))

    # assert len(connect_nodes_0)==len(connect_nodes_1), f"{len(connect_nodes_0)=}, {len(connect_nodes_1)=}"

    for nodes_0, nodes_1 in zip(connect_nodes_0, connect_nodes_1):
        nodes_1[0].connect_to(nodes_0[1])
        nodes_0[0].connect_to(nodes_1[1])

def _get_to_main_node(node, nodes_in_main, main_road, from_begin_road):
    # find nearest parent in main
    cur = node
    dis = 0
    while cur not in nodes_in_main:
        dis += 1
        cur = from_begin_road[cur]
    assert cur in nodes_in_main
    parent_in_main_road = cur

    dis *= 4
    

    parent_idx_in_main_road = None
    for idx, node in enumerate(main_road):
        if node==parent_in_main_road:
            parent_idx_in_main_road = idx
            break

    to_main_node = main_road[min(len(main_road)-1, parent_idx_in_main_road + dis)]

    print_rank_0(f"connect to main node: {node.name=}, {to_main_node.name=}, {dis=}")
    return to_main_node

def _get_from_main_node(node, nodes_in_main, main_road, to_end_road):
    # find nearest parent in main
    cur = node
    dis = 0
    while cur not in nodes_in_main:
        dis += 1
        cur = to_end_road[cur]
    assert cur in nodes_in_main
    child_in_main_road = cur

    dis *= 4

    child_idx_in_main_road = None
    for idx, node in enumerate(main_road):
        if node==child_in_main_road:
            child_idx_in_main_road = idx
            break

    from_main_node = main_road[max(0, child_idx_in_main_road - dis)]
    print_rank_0(f"connect from main node: {node.name=}, {from_main_node.name=}, {dis=}")
    return from_main_node


def main5():
    """
    
    """
    length = 100
    nodes0 = []
    last_comm = None
    for i in range(0,length):
        comp = HeavyComputationNode(i*2)
        comm = CommunicationNode(i*2+1)
        comp.set_users([comm])
        if last_comm is not None:
            last_comm.set_users([comp])
        last_comm = comm
        nodes0.append(comp)
        nodes0.append(comm)

    begin_val = length * 2
    nodes1 = []
    last_comp = None
    for i in range(0,length):
        comm = CommunicationNode(begin_val + i*5)
        comm2 = CommunicationNode(begin_val + i*5+1)
        comp1 = HeavyComputationNode(begin_val + i*5+2)
        comp2 = HeavyComputationNode(begin_val + i*5+3)
        comp3 = HeavyComputationNode(begin_val + i*5+4)
        
        comm.set_users([comp1, comm2])
        comm2.set_users([comp2])
        comp2.set_users([comp3])
        comp1.set_users([comp3])

        # comp.set_users([])
        if last_comp is not None:
            last_comp.set_users([comm])
        last_comp = comp3
        nodes1.append(comp1)
        nodes1.append(comp2)
        nodes1.append(comp3)
        nodes1.append(comm)
        nodes1.append(comm2)

    graph = Graph([*nodes0, *nodes1])
    print(f"{len(graph.nodes)}=")
    # exit()
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)

    for head in graph.forall_heads():
        head.show()
    
    
    print("-------- OverlapScheduler --------")
    scheduler = OverlapScheduler()
    begin_time = time.time()
    head_list = scheduler.schedule_overlap(graph)
    end_time = time.time()
    search_time = end_time - begin_time
    print("\nPartition:")
    for head in head_list:
        head.show()

    print(f"{scheduler._search_cnt=}")
    print(f"{search_time=} s")


def main6():
    """
    -
    """

    def build_chain(length):
        nodes = []
        last = None
        for i in range(length):
            comp = HeavyComputationNode(0)
            comm = CommunicationNode(0)

            if last is not None:
                last.set_users([comp])
            comp.set_users([comm])
            last = comm

            nodes.append(comp)
            nodes.append(comm)

        return nodes
            
    n = 16
    k = 4
    m = 2
    nodes = []
    edges = []

    last_end = None
    for i in range(n):
        chain_main = build_chain(k)
        chain_side = build_chain(m)
        chain_main[0].add_user(chain_side[0])
        chain_side[-1].add_user(chain_main[-1])

        if last_end is not None:
            last_end.add_user(chain_main[0])
        last_end = chain_main[-1]
        edges.append(last_end)
        nodes.extend(chain_main)
        nodes.extend(chain_side)

    last_end = None
    for i in range(n):
        chain_main = build_chain(k)
        chain_side = build_chain(1)
        chain_main[0].add_user(chain_side[0])
        chain_side[-1].add_user(chain_main[-1])

        if last_end is not None:
            last_end.add_user(chain_main[0])
        last_end = chain_main[-1]
        edges[i].add_user(last_end)
        nodes.extend(chain_main)
        nodes.extend(chain_side)

    graph = Graph(nodes)
    print(f"{len(graph.nodes)}=")
    # exit()
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)

    for head in graph.forall_heads():
        head.show()
    
    
    print("-------- OverlapScheduler --------")
    scheduler = OverlapScheduler()
    begin_time = time.time()
    head_list = scheduler.schedule_overlap(graph)
    end_time = time.time()
    search_time = end_time - begin_time
    print("\nPartition:")
    for head in head_list:
        head.show()

    print(f"{scheduler._search_cnt=}")
    print(f"{search_time=} s")

if __name__=="__main__":
    # import sys

    # # 获取当前的递归限制
    # current_limit = sys.getrecursionlimit()
    # print("Current recursion limit:", current_limit)

    # 设置递归限制为一个新值，例如设置为2000
    # new_limit = 2000
    # sys.setrecursionlimit(new_limit)
    # print("New recursion limit:", sys.getrecursionlimit())

    # cProfile.run('main4()')
    main6()