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

def move_item(obj, idx):
    a = obj[idx]
    del obj[idx]
    return a

def get_fake_tensor(node0):
    fake_tensor = node0.meta.get(
        "val", node0.meta.get("tensor_meta", node0.meta.get("example_value", None))
    )
    return fake_tensor

def unwrap(graph):
    placeholder_list = []
    for node in graph.nodes:
        if node.op=="placeholder":
            placeholder_list.append(node)
    
    for placeholder in placeholder_list:
        with graph.inserting_after(placeholder):
            new_node =  graph.call_function(move_item, args=(placeholder, 0))

        fake_tensor = get_fake_tensor(placeholder)
        if fake_tensor is not None:
            new_node.meta["val"] = fake_tensor.clone()

        for user in list(placeholder.users.keys()):
            if not user==new_node:
                _replace_args(user, placeholder, new_node)