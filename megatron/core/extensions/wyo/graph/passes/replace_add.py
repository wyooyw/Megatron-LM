import torch

def add(lhs, rhs):
    return lhs + rhs

def replace_add(graph):
    add_nodes = []
    for node in graph.nodes:
        if node.op=="call_function" and type(node.target) == torch._ops.OpOverload and node.target.__name__=="add.Tensor":
            add_nodes.append(node)
    for node in add_nodes:
        # print("hit replace add! ")
        with graph.inserting_before(node):
            new_node = graph.call_function(
                add, node.args, {}
            )
        replace_args_of_users(node, new_node)
        graph.erase_node(node)

def replace_args_of_users(old_node, new_node):
    users = list(old_node.users.keys())
    print(old_node, users)
    for user in users:
        replace_args(user, old_node, new_node)

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