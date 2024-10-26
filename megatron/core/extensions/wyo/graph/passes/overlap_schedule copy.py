import itertools
# import torch
from queue import Queue
from ordered_set import OrderedSet
import time
import cProfile
from megatron.core.extensions.wyo.graph.utils import print_rank_0
class Node:
    def __init__(self, val):
        self.val = val
        self.users = {}

    def set_users(self, nodes):
        self.users = {key:None for key in nodes}

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
                "all_reduce_in_tp_group"
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
                "mm",
            ]
        ):
            return True
        return False
    
    @staticmethod
    def is_output(node):
        return node.target == "output"

class Graph:
    def __init__(self, nodes):
        self.nodes = OrderedSet(nodes)
        self.graph_key = None

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

    def give_path_len_from_begin(self):
        """
        for each node, its path length from beign is
            maxPath(node) = max_{for arg in node.args} (maxPath(arg))
        """
        pass
        
        

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
        print(f"{len(zero_indegree_nodes)=}")

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
        print_rank_0("zero_indegree_comp_nodes:")
        for node in zero_indegree_comp_nodes:
            print_rank_0(f"  {node.op=}, {node.target=}")
        print_rank_0("zero_indegree_comm_nodes:")
        for node in zero_indegree_comm_nodes:
            print_rank_0(f"  {node.op=}, {node.target=}")

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
            head = self._enumerate_pure_computation_head_begin_with_node(node, node_indegree)
            yield PureComputionSubGraph(head)

    def _enumerate_pure_computation_head_begin_with_node(self, begin_node, node_indegree, must_contain_heavy_comp_node=False):
        # print("_enumerate_pure_computation_head_begin_with_node")
        head = []
        cur_node = begin_node
        contain_heavy_node = False
        while True:
            head.append(cur_node)

            if NodeClassifier.is_heavy_computation_node(cur_node):
                contain_heavy_node = True

            if (
                NodeClassifier.is_heavy_computation_node(cur_node)
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

        if contain_heavy_node and must_contain_heavy_comp_node:
            return None
        
        return head

    def _enumerate_compution_communication_overlap_head(self, zero_indegree_comp_nodes, zero_indegree_comm_nodes):
        if len(zero_indegree_comp_nodes) > 4:
            return []
        
        node_indegree = self.get_node_indegree()
        
        # TODO: purn this space
        comp_heads = []
        for comp_node in zero_indegree_comp_nodes:
            comp_head = self._enumerate_pure_computation_head_begin_with_node(
                begin_node=comp_node,
                node_indegree=node_indegree
            )
            comp_heads.append(comp_head)
        # print(f"{len(comp_heads)=}")
        # pick one comm_node
        for comm_node in zero_indegree_comm_nodes:
            
            # try all subset of comp_nodes
            # for subset_len in range(1, len(comp_heads) + 1):
            for subset_len in range(1, min(len(comp_heads) + 1, 2)):
                for comp_heads_subset in itertools.combinations(comp_heads, subset_len):
                    comp_nodes = [item for sublist in comp_heads_subset for item in sublist]
                    yield ComputationCommunicationOverlapSubGraph(comm_node=comm_node, comp_nodes=comp_nodes)

class ComputationCommunicationOverlapSubGraph:
    def __init__(self, comm_node, comp_nodes):
        assert comp_nodes is None or type(comp_nodes)==list
        self.comm_node = comm_node
        self.comp_nodes = comp_nodes

    def _show(self):
        if self.comp_nodes is not None:
            comp_nodes_val = [f"({node.op})" for node in self.comp_nodes]
            comp_nodes_idx = [node.__sort_idx__ for node in self.comp_nodes]
            print_rank_0(f"    comp nodes: val={comp_nodes_val}, idx={comp_nodes_idx}")
        
        if self.comm_node is not None:
            print_rank_0(f"    comm nodes: {self.comm_node.op}, idx={self.comm_node.__sort_idx__}")

    def show(self):
        print_rank_0("comm comp overlap head:")
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
        print_rank_0("comp head:")
        self._show()
            
            
class PureCommunicationSubGraph(ComputationCommunicationOverlapSubGraph):
    def __init__(self, comm_node):
        super().__init__(comm_node, None)

    def show(self):
        print_rank_0("comm head:")
        self._show()



def get_output_node(graph):
    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break
    return output_node


class OverlapScheduler:
    def __init__(self):
        self.record_score = dict()
        self.record_head = dict()
    
    def record_result(self, graph, head, score):
        graph_key = graph.get_key()
        self.record_score[graph_key] = score
        self.record_head[graph_key] = head

    def get_score(self, head):
        return head.score()

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
            print(f"{self._search_cnt=}")

        min_score = 9999999999
        min_head = None
        for head in graph.forall_heads():
            head_score = self.get_score(head)
            left_graph = graph.erase(head.get_nodes())
            left_graph_score = self.find_best_partition_strategy(left_graph)
            # print(f"{head_score=}, {left_graph_score=}")
            if head_score + left_graph_score < min_score:
                min_score = head_score + left_graph_score
                min_head = head
                
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
        # from megatron.core.extensions.wyo.model.communicate.communicate import wait_tensor
        print("schedule_overlap begin")
        self.find_best_partition_strategy(graph)
        print("find_best_partition_strategy finish")
        # print("")
        # print(f"{self.record_score=}")
        # print("")
        # print(f"{self.record_head=}")
        # exit()
        head_list = self.get_partition_from_record_head(graph)
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
            async_comm_node = graph.call_function(node, args=args, kwargs={"async_op": True})
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
    for node in graph.nodes:
        if not (NodeClassifier.is_output(node) or NodeClassifier.is_placeholder_node(node)):
            nodes.append(node)
        
    graph = Graph(nodes)
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)

    print_rank_0("\n----------- get_node_indegree ----------")
    node_indegree = graph.get_node_indegree()
    for node, indegree in node_indegree.items():
        print_rank_0(f"{node.op=}, {node.target=}, {indegree=}")
    exit()

    # for head in graph.forall_heads():
    #     head.show()

    # scheduler = OverlapScheduler()
    # begin_time = time.time()
    # fused_graph = scheduler.schedule_overlap(graph)
    # end_time = time.time()
    
    # search_time = end_time - begin_time
    # print(f"{scheduler._search_cnt=}")
    # print(f"{search_time=} s")
    exit()
    return fused_graph

def main1():
    """
    ComputationNode
    CommunicationNode
    HeavyComputationNode
    """
    comp0 = ComputationNode(0)
    comp1 = ComputationNode(1)
    comp2 = HeavyComputationNode(2)
    comp3 = ComputationNode(3)
    comp4 = HeavyComputationNode(4)

    comp0.set_users([comp1])
    comp1.set_users([comp2])
    comp2.set_users([comp3])
    comp3.set_users([comp4])

    graph = Graph([comp0,comp1,comp2,comp3,comp4])
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)

    for head in graph.forall_heads():
        head.show()


def main2():
    """
    ComputationNode
    CommunicationNode
    HeavyComputationNode
    """
    comp0 = ComputationNode(0)
    comp1 = ComputationNode(1)
    comp2 = ComputationNode(2)
    comp3 = ComputationNode(3)
    comp4 = ComputationNode(4)
    comp5 = ComputationNode(5)
    # comp6 = ComputationNode(6)
    # comp7 = ComputationNode(7)

    comp0.set_users([comp1, comp3])
    comp1.set_users([comp2])
    comp2.set_users([comp4])
    comp3.set_users([comp4])
    comp4.set_users([comp5])
    # comp4.set_users([comp5])
    # comp5.set_users([comp6])

    graph = Graph([comp0,comp1,comp2,comp3,comp4, comp5])
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)
    # exit()
    for head in graph.forall_heads():
        head.show()
    
    print("-------- OverlapScheduler --------")
    scheduler = OverlapScheduler()
    head_list = scheduler.schedule_overlap(graph)
    print("\nPartition:")
    for head in head_list:
        head.show()

def main3():
    """
    带分支：
    2: 43
    3: 196
    4: 780
    5: 2847
    6: 9887
    7: 33304
    8: 110124

    纯线性：
    8: 101
    100: 12751
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
        comm = CommunicationNode(begin_val + i*3)
        comp1 = HeavyComputationNode(begin_val + i*3+1)
        comp2 = ComputationNode(begin_val + i*3+2)
        comm.set_users([comp1])
        comp1.set_users([comp2])

        if last_comp is not None:
            last_comp.set_users([comm])
        last_comp = comp2

        nodes1.append(comm)
        nodes1.append(comp1)
        nodes1.append(comp2)

    graph = Graph([*nodes0, *nodes1])
    print(f"{len(graph.nodes)}=")
    # exit()
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)
    # exit()
    for head in graph.forall_heads():
        head.show()
    
    print("-------- OverlapScheduler --------")
    scheduler = OverlapScheduler()
    head_list = scheduler.schedule_overlap(graph)
    print("\nPartition:")
    for head in head_list:
        head.show()

    print(f"{scheduler._search_cnt=}")


def main4():
    """
    
    """
    length = 20
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
        comm = CommunicationNode(begin_val + i*3)
        comp = HeavyComputationNode(begin_val + i*3+1)
        comp2 = ComputationNode(begin_val + i*3+2)
        
        comm.set_users([comp, comp2])
        # comp.set_users([])
        if last_comp is not None:
            last_comp.set_users([comm])
        last_comp = comp
        nodes1.append(comp)
        nodes1.append(comp2)
        nodes1.append(comm)

    graph = Graph([*nodes0, *nodes1])
    print(f"{len(graph.nodes)}=")
    # exit()
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)
    # exit()
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


def main5():
    """
    
    """
    length = 10
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
        comm = CommunicationNode(begin_val + i*6)
        comp1 = ComputationNode(begin_val + i*6+1)
        comp2 = ComputationNode(begin_val + i*6+2)
        comp3 = ComputationNode(begin_val + i*6+3)
        comp4 = ComputationNode(begin_val + i*6+4)
        comp5 = HeavyComputationNode(begin_val + i*6+5)
        
        comm.set_users([comp, comp1, comp2, comp3, comp4])
        comp1.set_users([comp5])
        comp2.set_users([comp5])
        comp3.set_users([comp5])
        comp4.set_users([comp5])

        # comp.set_users([])
        if last_comp is not None:
            last_comp.set_users([comm])
        last_comp = comp5
        nodes1.append(comp1)
        nodes1.append(comp2)
        nodes1.append(comp3)
        nodes1.append(comp4)
        nodes1.append(comp5)
        nodes1.append(comm)

    graph = Graph([*nodes0, *nodes1])
    print(f"{len(graph.nodes)}=")
    # exit()
    graph.topo_sort_and_give_sort_idx()
    graph.show(sorted=True)

    print("\n----------- get_node_indegree ----------")
    node_indegree = graph.get_node_indegree()
    for node, indegree in node_indegree.items():
        print(f"{node.op=}, {node.target=}, {indegree=}")
    exit()
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
    
    # cProfile.run('main4()')
    main5()