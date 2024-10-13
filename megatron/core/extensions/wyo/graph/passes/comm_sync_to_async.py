import torch
from megatron.core.extensions.wyo.model.communicate.communicate import wait_tensor

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

def _sync_to_async_by_set_kwarg_and_add_wait_op(graph, comm_op_name):
    comm_nodes = []
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "op")
            and node.target._opname == comm_op_name
            and (not ("asnyc_op" in node.kwargs and node.kwargs["async_op"]==True))
        ):
            comm_nodes.append(node)

    for node in comm_nodes:
        print(f"{node.args=}, {node.kwargs=}")
        # set kwarg
        # node.update_kwarg("asnyc_op", True)
        
        # create wait nodes
        with graph.inserting_before(node):
            new_node = graph.call_function(node.target, args=node.args, kwargs={"async_op": True})
            node_wait = graph.call_function(wait_tensor, args=(new_node,))
            # node_clone = graph.call_function(torch.ops.aten.clone.default, args=(node_wait, ))

        # replace
        users = list(node.users.keys())
        for user in users:
            _replace_args(user, node, node_wait)
        assert len(node.users) == 0, f"{node.users=}"
        assert len(node_wait.users) == len(
            users
        ), f"{len(node_wait.users)=}, {len(users)=}"
        graph.erase_node(node)
    
    print(f"replace {len(comm_nodes)} {comm_op_name} op into async!")

def comm_sync_to_async(graph):
    # pass
    _sync_to_async_by_set_kwarg_and_add_wait_op(graph, "gather_along_first_dim_in_tp_group")
    _sync_to_async_by_set_kwarg_and_add_wait_op(graph, "reduce_scatter_along_first_dim_in_tp_group")