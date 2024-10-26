import torch

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
    
def redundant_clone_eliminate(graph):
    """
    A -> clone -> B
    if A has only one user(clone)
    then set A to clone's user, delete clone
    """
    clone_nodes = []

    for node in graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "op")
            and node.target._opname == "clone"
        ):
            clone_nodes.append(node)
    print(f"{len(clone_nodes)=}")

    num_eliminate_clone_node = 0
    for clone_node in clone_nodes:
        args = clone_node.args
        if "memory_format" in clone_node.kwargs and clone_node.kwargs["memory_format"]==torch.contiguous_format:
            continue

        assert len(args)==1, f"{args=}"
        parent = args[0]

        if len(parent.users)>1:
            continue
        
        # eliminate this clone!
        users = list(clone_node.users.keys())
        for user in users:
            _replace_args(user, clone_node, parent)

        assert len(clone_node.users) == 0, f"{node.users=}"
        assert len(parent.users) == len(users) + 1, f"{len(parent.users)=}, {len(users)=}"
        graph.erase_node(clone_node)
        num_eliminate_clone_node += 1


    print(f"{num_eliminate_clone_node=}")