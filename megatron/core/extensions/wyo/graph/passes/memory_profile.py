import torch
from megatron.core.extensions.wyo.graph.utils import print_rank_0
import gc
import time
def memory_profile(id):
    max_memory = torch.cuda.max_memory_allocated()
    max_memory = max_memory / 1024 / 1024 / 1024
    print_rank_0(f"memory_profile at {id=}: {max_memory=:.3f}GB")

def memory_collect():
    # gc.collect()
    time.sleep(0.02)
    pass

def insert_memory_profile(graph, round=10):
    for idx,node in enumerate(graph.nodes):
        if (idx % round)==0:
            with graph.inserting_before(node):
                new_node = graph.call_function(memory_collect)
                # new_node = graph.call_function(memory_profile, kwargs={"id": idx})