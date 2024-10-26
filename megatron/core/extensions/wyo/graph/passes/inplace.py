from megatron.core.extensions.wyo.model.communicate.communicate import all_reduce_in_tp_group_inplace
from megatron.core.extensions.wyo.graph.graph_utils import get_fake_tensor
from megatron.core.extensions.wyo.model.submodule.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy_backward_inplace

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

def inplace_allreduce(graph):
    """
    Make all-reduce op inplace
    """
    allreduce_nodes = []
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "op")
            and node.target._opname == "all_reduce_in_tp_group"
        ):
            allreduce_nodes.append(node)

    for node in allreduce_nodes:
        print(f"{node.args=}, {node.kwargs=}")
        # set kwarg
        # node.update_kwarg("asnyc_op", True)
        
        # create wait nodes
        with graph.inserting_before(node):
            new_node = graph.call_function(all_reduce_in_tp_group_inplace, args=node.args, kwargs=node.kwargs)

        new_node.meta["val"] = get_fake_tensor(node).clone()

        # replace
        users = list(node.users.keys())
        for user in users:
            _replace_args(user, node, new_node)
        assert len(node.users) == 0, f"{node.users=}"
        assert len(new_node.users) == len(
            users
        ), f"{len(new_node.users)=}, {len(users)=}"
        graph.erase_node(node)

def inplace_vocab_backward(graph):
    """
    Make all-reduce op inplace
    """
    vocab_backward_nodes = []
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "vocab_parallel_cross_entropy_backward.default"
        ):
            vocab_backward_nodes.append(node)

    for node in vocab_backward_nodes:
        print(f"{node.args=}, {node.kwargs=}")
        # set kwarg
        # node.update_kwarg("asnyc_op", True)
        
        # create wait nodes
        with graph.inserting_before(node):
            new_node = graph.call_function(vocab_parallel_cross_entropy_backward_inplace, args=node.args, kwargs=node.kwargs)

        new_node.meta["val"] = get_fake_tensor(node).clone()

        # replace
        users = list(node.users.keys())
        for user in users:
            _replace_args(user, node, new_node)
        assert len(node.users) == 0, f"{node.users=}"
        assert len(new_node.users) == len(
            users
        ), f"{len(new_node.users)=}, {len(users)=}"
        graph.erase_node(node)


def inplace(graph):
    inplace_allreduce(graph)
    inplace_vocab_backward(graph)