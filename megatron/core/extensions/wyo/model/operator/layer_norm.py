from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
import torch
import importlib
from typing import Tuple, List

fused_layer_norm_cuda = None

@torch.library.custom_op("wrapped_ops::layer_norm", mutates_args=())
def layer_norm(input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                normalized_shape: int,
                eps: float,
                memory_efficient: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        assert input.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()
        input_ = input
        weight_ = weight
        bias_ = bias
        normalized_shape = (normalized_shape,)
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(
            input_, normalized_shape, weight_, bias_, eps
        )
        # if memory_efficient:
        #     ctx.save_for_backward(output, weight_, bias_, None, invvar)
        # else:
        #     ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output, mean, invvar

@layer_norm.register_fake
def layer_norm_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalized_shape: int,
    eps: float,
    memory_efficient: bool = False
):
    output = input.new_empty(*input.shape)
    # TODO: force fp32?
    mean_dim = len(input.shape) - 1
    mean = input.new_empty(*input.shape[:mean_dim])
    invvar = input.new_empty(*input.shape[:mean_dim])
    
    return output, mean, invvar

@torch.library.custom_op("wrapped_ops::layer_norm_bwd", mutates_args=())
def layer_norm_bwd(
    grad_output: torch.Tensor,
    input_or_output: torch.Tensor,
    weight_: torch.Tensor,
    bias_: torch.Tensor,
    mean: torch.Tensor,
    invvar: torch.Tensor,
    normalized_shape: int,
    eps: float,
    memory_efficient: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # input_or_output, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    normalized_shape = (normalized_shape, )
    grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar, input_or_output,
        normalized_shape, weight_, bias_, eps, memory_efficient
    )
    return grad_input, grad_weight, grad_bias
     
@layer_norm_bwd.register_fake
def layer_norm_bwd_abstract(
    grad_output: torch.Tensor,
    input_or_output: torch.Tensor,
    weight_: torch.Tensor,
    bias_: torch.Tensor,
    mean: torch.Tensor,
    invvar: torch.Tensor,
    normalized_shape: int,
    eps: float,
    memory_efficient: bool
):
    return grad_output.new_empty(*grad_output.shape), weight_.new_empty(weight_.shape), bias_.new_empty(bias_.shape)

def _backward_layer_norm(ctx, dout, dmean, dvar):
    input_or_output, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input, grad_weight, grad_bias = layer_norm_bwd(
        dout,
        input_or_output,
        weight_,
        bias_,
        mean,
        invvar,
        ctx.normalized_shape,
        ctx.eps,
        ctx.memory_efficient
    )
    return grad_input, grad_weight, grad_bias, None, None, None

def setup_context(ctx, inputs, output):
    (
        input,
        weight,
        bias,
        normalized_shape,
        eps,
        memory_efficient
    ) = inputs
    
    (output, mean, invvar) = output

    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    ctx.memory_efficient = memory_efficient

    if memory_efficient:
        ctx.save_for_backward(output, weight, bias, None, invvar)
    else:
        ctx.save_for_backward(input, weight, bias, mean, invvar)
    

layer_norm.register_autograd(_backward_layer_norm, setup_context=setup_context)
    # @staticmethod
    # def backward(ctx, grad_output):
    #     input_or_output, weight_, bias_, mean, invvar = ctx.saved_tensors
    #     grad_input = grad_weight = grad_bias = None
    #     grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
    #         grad_output.contiguous(), mean, invvar, input_or_output,
    #         ctx.normalized_shape, weight_, bias_, ctx.eps, ctx.memory_efficient
    #     )
    #     return grad_input, grad_weight, grad_bias, None, None, None


def main():
    hidden = 4

    input = torch.randn((1, 4, hidden), device="cuda")
    input2 = input.detach().clone()
    weight = torch.nn.Parameter(torch.randn((hidden,), device="cuda"))
    weight2 = torch.nn.Parameter(weight.detach().clone())
    bias = torch.nn.Parameter(torch.randn((hidden,), device="cuda"))
    bias2 = torch.nn.Parameter(bias.detach().clone())

    layernorm_output = FusedLayerNormAffineFunction.apply(
        input, 
        weight, # weight 
        bias,   # bias
        (hidden,), 
        1e-4, 
        True
    )
    layernorm_output.sum().backward()

    layernorm_output_2 = layer_norm(
        input2,
        weight2,
        bias2,
        hidden,
        1e-4,
        True
    )
    layernorm_output_2[0].sum().backward()

    print(layernorm_output)
    print(layernorm_output_2)
    
    print(f"{weight.grad=}")
    print(f"{weight2.grad=}")

    print(f"{bias.grad=}")
    print(f"{bias2.grad=}")

if __name__=="__main__":
    main()