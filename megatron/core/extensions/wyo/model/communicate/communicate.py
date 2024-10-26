import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.core.extensions.wyo.status import is_current_status_trace

def no_setup(ctx, inputs, output):
    pass

HANDLES = dict()

@torch.library.custom_op("wrapped_ops::gather_along_first_dim_in_tp_group", mutates_args=[])
def gather_along_first_dim_in_tp_group(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    output, handle = _gather_along_first_dim(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=async_op
    )
    if async_op:
        HANDLES[output] = handle
    return output

@gather_along_first_dim_in_tp_group.register_fake
def gather_along_first_dim_in_tp_group_abstract(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    dim_size = list(_inp.size())
    dim_size[0] = dim_size[0] * world_size
    return _inp.new_empty(dim_size)

def _backward_gather_along_first_dim_in_tp_group(ctx, dout):
    din = reduce_scatter_along_first_dim_in_tp_group(dout)
    return din, None

gather_along_first_dim_in_tp_group.register_autograd(_backward_gather_along_first_dim_in_tp_group, setup_context=no_setup)

class GatherAlongFirstDimInTpGroup(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _inp: torch.Tensor
    ):
        output, _ = _gather_along_first_dim(
            _inp, 
            process_group=parallel_state.get_tensor_model_parallel_group(),
            async_op=False
        )
        return output
    
    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor
    ):
        din = _reduce_scatter_along_first_dim(
            dout,
            process_group=parallel_state.get_tensor_model_parallel_group(),
            async_op=False
        )
        return din

"""
Reduce-Scatter along first dim in tp group
"""

@torch.library.custom_op("wrapped_ops::reduce_scatter_along_first_dim_in_tp_group", mutates_args=[])
def reduce_scatter_along_first_dim_in_tp_group(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    output, handle = _reduce_scatter_along_first_dim(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=async_op
    )
    if async_op:
        HANDLES[output] = handle
    return output

@reduce_scatter_along_first_dim_in_tp_group.register_fake
def reduce_scatter_along_first_dim_in_tp_group_abstract(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    dim_size = list(_inp.size())
    dim_size[0] = dim_size[0] // world_size
    return _inp.new_empty(dim_size)

def _backward_reduce_scatter_along_first_dim_in_tp_group(ctx, dout):
    din = gather_along_first_dim_in_tp_group(dout)
    return din, None

reduce_scatter_along_first_dim_in_tp_group.register_autograd(_backward_reduce_scatter_along_first_dim_in_tp_group, setup_context=no_setup)

"""
Scatter along first dim in tp group
"""

@torch.library.custom_op("wrapped_ops::scatter_along_first_dim_in_tp_group", mutates_args=[])
def scatter_along_first_dim_in_tp_group(_inp: torch.Tensor) -> torch.Tensor:
    result = _split_along_first_dim(_inp)
    if is_current_status_trace():
        result = result.clone()
    
    return result


@scatter_along_first_dim_in_tp_group.register_fake
def scatter_along_first_dim_in_tp_group_abstract(_inp: torch.Tensor) -> torch.Tensor:
    return _split_along_first_dim(_inp).clone()

def _backward_scatter_along_first_dim_in_tp_group(ctx, dout):
    return gather_along_first_dim_in_tp_group(dout)

scatter_along_first_dim_in_tp_group.register_autograd(_backward_scatter_along_first_dim_in_tp_group, setup_context=no_setup)

"""
AllReduce in tp group
"""

@torch.library.custom_op("wrapped_ops::all_reduce_in_tp_group", mutates_args=[])
def all_reduce_in_tp_group(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    output, handle = _all_reduce(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=async_op
    )

    # make dynamo happy
    output = output.clone()
    
    if async_op:
        HANDLES[output] = handle

    return output

@all_reduce_in_tp_group.register_fake
def all_reduce_in_tp_group_abstract(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    return _inp.new_empty(list(_inp.size()))

def _backward_all_reduce_in_tp_group(ctx, dout):
    din = all_reduce_in_tp_group(dout)
    return din, None

all_reduce_in_tp_group.register_autograd(_backward_all_reduce_in_tp_group, setup_context=no_setup)

def all_reduce_in_tp_group_inplace(_inp: torch.Tensor, async_op: bool=False) -> torch.Tensor:
    output, handle = _all_reduce(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=async_op
    )
    if async_op:
        HANDLES[output] = handle

    return output

"""
Wait
"""

def wait_tensor(tensor: torch.Tensor) -> None:
    handle = HANDLES[tensor]
    handle.wait()
    del HANDLES[tensor]
    return tensor

"""
Basic functions
"""

def get_distributed_world_size(group=None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)

def _gather_along_first_dim(
    input_: torch.Tensor,
    process_group,
    async_op: bool = False,
    output=None
) -> Tuple[torch.Tensor, Any]:
    """All-gather tensors and concatenate along first dimension."""

    # Return immediately if no communication is required
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return input_, None

    # Allocate output tensor
    output_shape = list(input_.size())
    output_shape[0] *= world_size

    if output is None:
        output = torch.empty(
            output_shape,
            dtype=input_.dtype,
            device=input_.device,
            memory_format=torch.contiguous_format,
        )
    
    src = input_.contiguous()
    dst = output

    # Launch all-gather
    handle = torch.distributed.all_gather_into_tensor(
        dst,
        src,
        group=process_group,
        async_op=async_op,
    )
    return output, handle

def _reduce_scatter_along_first_dim(
    input_: torch.Tensor, process_group, async_op: bool = False, output=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_distributed_world_size(process_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    if output is None:
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle

def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = parallel_state.get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = parallel_state.get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output

def _all_reduce(input_, process_group, async_op=False):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    world_size = get_distributed_world_size(process_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    # All-reduce.
    handle = torch.distributed.all_reduce(input_.contiguous(), group=process_group, async_op=async_op)
        
    return input_, handle

def reduce_scatter_along_first_dim_in_tp_group_inplace(_inp: torch.Tensor, async_op: bool=False, out: torch.Tensor=None) -> torch.Tensor:
    output, handle = _reduce_scatter_along_first_dim(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=async_op,
        output=out
    )
    if async_op:
        HANDLES[output] = handle
    return output

def gather_along_first_dim_in_tp_group_inplace(_inp: torch.Tensor, async_op: bool=False, out: torch.Tensor=None) -> torch.Tensor:
    output, handle = _gather_along_first_dim(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=async_op,
        output=out
    )
    if async_op:
        HANDLES[output] = handle
    return output