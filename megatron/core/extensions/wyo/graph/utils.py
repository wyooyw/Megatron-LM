import random
import torch
import torch.distributed as dist

def seed_everything(seed=11):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_rank_0():
    return (not dist.is_initialized()) or dist.get_rank()==0

def print_rank_0(text):
    if is_rank_0():
        print(f"{text}")

def print_graph_rank_0(graph):
    python_code = graph.python_code("module", verbose=True)
    code = python_code.src
    print_rank_0(code)