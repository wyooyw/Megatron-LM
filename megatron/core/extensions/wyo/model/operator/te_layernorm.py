


from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
import torch
import importlib
from typing import Tuple, List
from transformer_engine.pytorch.module._common import _apply_normalization
import transformer_engine.pytorch.cpp_extensions as tex

fused_layer_norm_cuda = None

@torch.library.custom_op("wrapped_ops::layer_norm", mutates_args=())
def layer_norm(input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                normalized_shape: int,
                eps: float,
                memory_efficient: bool
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # assert len(input.shape)==2
    ln_out = torch.empty_like(
        input, dtype=input.dtype, memory_format=torch.contiguous_format
    )
    origin_shape = input.shape
    input = input.reshape(-1, input.shape[-1])
    ln_out, mu, rsigma = _apply_normalization(
        input,
        ln_out,
        weight,
        bias,
        eps,
        False,
        False,
        "LayerNorm",
        0, # fwd_ln_sm_margin
        False, # zero_centered_gamma,
        True, #is_grad_enabled,
    )
    ln_out = ln_out.reshape(*origin_shape)
    return ln_out, mu, rsigma

@layer_norm.register_fake
def layer_norm_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalized_shape: int,
    eps: float,
    memory_efficient: bool
):

    output = input.new_empty(*input.shape)
    # TODO: force fp32?
    size = input.reshape(-1, input.shape[-1]).shape[0]

    mu = torch.empty((size,), dtype=torch.float32)
    rsigma = torch.empty((size,), dtype=torch.float32)
    
    return output, mu, rsigma

@torch.library.custom_op("wrapped_ops::layer_norm_bwd", mutates_args=())
def layer_norm_bwd(
    grad_output: torch.Tensor,
    inputmat: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    normalized_shape: int,
    eps: float,
    memory_efficient: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # input_or_output, weight_, bias_, mean, invvar = ctx.saved_tensors

    origin_shape = inputmat.shape
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    inputmat = inputmat.reshape(-1, inputmat.shape[-1])

    dgrad, dgamma, dbeta = tex.layernorm_bwd(
        grad_output,
        inputmat,
        mu,
        rsigma,
        weight,
        0, # bwd_ln_sm_margin,
        False, # ctx.zero_centered_gamma,
    )

    dgrad = dgrad.reshape(*origin_shape)
    
    return dgrad, dgamma, dbeta
     
@layer_norm_bwd.register_fake
def layer_norm_bwd_abstract(
    grad_output: torch.Tensor,
    inputmat: torch.Tensor,
    weight_: torch.Tensor,
    bias_: torch.Tensor,
    mean: torch.Tensor,
    invvar: torch.Tensor,
    normalized_shape: int,
    eps: float,
    memory_efficient: bool
):
    return grad_output.new_empty(*grad_output.shape), weight_.new_empty(weight_.shape), bias_.new_empty(bias_.shape)

def _backward_layer_norm(ctx, dout, dmu, drsigma):
    inputmat, weight_, bias_, mu, rsigma = ctx.saved_tensors
    grad_input, grad_weight, grad_bias = layer_norm_bwd(
        dout,
        inputmat,
        weight_,
        bias_,
        mu,
        rsigma,
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
    
    (output, mu, rsigma) = output

    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    ctx.memory_efficient = memory_efficient

    # if memory_efficient:
    #     ctx.save_for_backward(output, weight, bias, None, invvar)
    # else:
    ctx.save_for_backward(input, weight, bias, mu, rsigma)
    

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

def layer_norm_inplace(input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                normalized_shape: int,
                eps: float,
                memory_efficient: bool,
                out: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # assert len(input.shape)==2
    # ln_out = torch.empty_like(
    #     input, dtype=input.dtype, memory_format=torch.contiguous_format
    # )
    ln_out = out
    origin_shape = input.shape
    input = input.reshape(-1, input.shape[-1])
    ln_out, mu, rsigma = _apply_normalization(
        input,
        ln_out,
        weight,
        bias,
        eps,
        False,
        False,
        "LayerNorm",
        0, # fwd_ln_sm_margin
        False, # zero_centered_gamma,
        True, #is_grad_enabled,
    )
    ln_out = ln_out.reshape(*origin_shape)
    return ln_out, mu, rsigma


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