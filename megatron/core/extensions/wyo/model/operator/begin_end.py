import torch
from megatron.core.extensions.wyo.status import is_current_status_trace

@torch.library.custom_op("wrapped_ops::begin", mutates_args=())
def begin(input: torch.Tensor) -> torch.Tensor:
    if is_current_status_trace():
        input = input.clone()
    return input

@begin.register_fake
def begin_abstract(input: torch.Tensor) -> torch.Tensor:
    return input.new_empty(list(input.size()))

@torch.library.custom_op("wrapped_ops::end", mutates_args=())
def end(input: torch.Tensor) -> torch.Tensor:
    if is_current_status_trace():
        input = input.clone()
    return input

@end.register_fake
def end_abstract(input: torch.Tensor) -> torch.Tensor:
    return input.new_empty(list(input.size()))

def _backward_begin(ctx, grad_output):
    return end(grad_output)

def _backward_end(ctx, grad_output):
    return begin(grad_output)

def setup_context(ctx, inputs, output):
    pass

begin.register_autograd(_backward_begin, setup_context=setup_context)
end.register_autograd(_backward_end, setup_context=setup_context)
