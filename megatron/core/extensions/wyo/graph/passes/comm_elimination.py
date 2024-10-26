import torch
from megatron.core.extensions.wyo.graph.utils import print_rank_0, is_rank_0

def _replace_args(node, old_arg, new_arg):
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
    
def comm_elimination(graph):
    """
    remove all communication nodes
    just for test
    """
    comm_nodes = []

    for node in graph.nodes:
        if (
            node.op == "call_function"
        ):
            print_rank_0(f"comm_elimination {node.target=}")
            # fn_name = getattr(node.target, "__name__")
            # return fn_name in ["wait_tensor", "wait_tensor.default"]
            if ( hasattr(node.target, "__name__") and getattr(node.target, "__name__") in [
                "all_reduce_in_tp_group_inplace",
            ]):
                comm_nodes.append(node)

    num_eliminate_comm_node = 0
    for comm_node in comm_nodes:
        args = comm_node.args
        assert len(args)==1
        parent = args[0]
        
        # eliminate this clone!
        comm_users = list(comm_node.users.keys())
        assert len(comm_users)==1
        wait_node = comm_users[0]

        wait_users = list(wait_node.users.keys())
        for user in wait_users:
            _replace_args(user, wait_node, parent)

        assert len(wait_node.users) == 0, f"{wait_node.users=}"
        # assert len(parent.users) == len(users) + 1, f"{len(parent.users)=}, {len(users)=}"
        graph.erase_node(wait_node)
        graph.erase_node(comm_node)
        num_eliminate_comm_node += 1


    print(f"{num_eliminate_comm_node=}")