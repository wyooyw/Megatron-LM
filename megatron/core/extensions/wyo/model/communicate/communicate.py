import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from megatron.core import parallel_state

def no_setup(ctx, inputs, output):
    pass


@torch.library.custom_op("wrapped_ops::gather_along_first_dim_in_tp_group", mutates_args=[])
def gather_along_first_dim_in_tp_group(_inp: torch.Tensor) -> torch.Tensor:
    output, _ = _gather_along_first_dim(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=False
    )
    return output

@gather_along_first_dim_in_tp_group.register_fake
def gather_along_first_dim_in_tp_group_abstract(_inp: torch.Tensor) -> torch.Tensor:
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    dim_size = list(_inp.size())
    dim_size[0] = dim_size[0] * world_size
    return _inp.new_empty(dim_size)

def _backward_gather_along_first_dim_in_tp_group(ctx, dout):
    din = reduce_scatter_along_first_dim_in_tp_group(dout)
    return din

gather_along_first_dim_in_tp_group.register_autograd(_backward_gather_along_first_dim_in_tp_group, setup_context=no_setup)


"""
Reduce-Scatter along first dim in tp group
"""

@torch.library.custom_op("wrapped_ops::reduce_scatter_along_first_dim_in_tp_group", mutates_args=[])
def reduce_scatter_along_first_dim_in_tp_group(_inp: torch.Tensor) -> torch.Tensor:
    output, _ = _reduce_scatter_along_first_dim(
        _inp, 
        process_group=parallel_state.get_tensor_model_parallel_group(),
        async_op=False
    )
    return output

@reduce_scatter_along_first_dim_in_tp_group.register_fake
def reduce_scatter_along_first_dim_in_tp_group_abstract(_inp: torch.Tensor) -> torch.Tensor:
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    dim_size = list(_inp.size())
    dim_size[0] = dim_size[0] // world_size
    return _inp.new_empty(dim_size)

def _backward_reduce_scatter_along_first_dim_in_tp_group(ctx, dout):
    din = gather_along_first_dim_in_tp_group(dout)
    return din

reduce_scatter_along_first_dim_in_tp_group.register_autograd(_backward_reduce_scatter_along_first_dim_in_tp_group, setup_context=no_setup)

"""
Scatter along first dim in tp group
"""

@torch.library.custom_op("wrapped_ops::scatter_along_first_dim_in_tp_group", mutates_args=[])
def scatter_along_first_dim_in_tp_group(_inp: torch.Tensor) -> torch.Tensor:
    return _split_along_first_dim(_inp).clone()

@scatter_along_first_dim_in_tp_group.register_fake
def scatter_along_first_dim_in_tp_group_abstract(_inp: torch.Tensor) -> torch.Tensor:
    return _split_along_first_dim(_inp).clone()

def _backward_scatter_along_first_dim_in_tp_group(ctx, dout):
    return gather_along_first_dim_in_tp_group(dout)

scatter_along_first_dim_in_tp_group.register_autograd(_backward_scatter_along_first_dim_in_tp_group, setup_context=no_setup)

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
) -> Tuple[torch.Tensor, Any]:
    """All-gather tensors and concatenate along first dimension."""

    # Return immediately if no communication is required
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return input_, None

    # Allocate output tensor
    output_shape = list(input_.size())
    output_shape[0] *= world_size
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
    input_: torch.Tensor, process_group, async_op: bool = False
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

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
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