import torch

import torch.nn as nn
# from flash_attn import _flash_attn_forward, _flash_attn_backward
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward

from typing import Sequence
from typing import Tuple
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnMaskType,
)
import flash_attn_2_cuda as flash_attn_cuda
from megatron.core.extensions.wyo.grad_clip import GradClip

# Use torch.library.custom_op to define a new custom operator.
# If your operator mutates any input Tensors, their names must be specified
# in the ``mutates_args`` argument.
@torch.library.custom_op("wrapped_ops::flash_attn", mutates_args=())
def flash_attn(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                dropout_p: float,
                softmax_scale: float,
                causal: bool,
                deterministic: bool,
                return_softmax: bool
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = None
    softmax_lse = None
    S_dmask = None 
    out_padded = None
    rng_state = None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # print(f"wyo flash_attn {softmax_scale=}, {causal=}, window_size=[-1,0] ")
    # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
    #     q,
    #     k,
    #     v,
    #     dropout_p,
    #     softmax_scale,
    #     causal=causal,
    #     window_size=(-1, -1),
    #     # softcap=0.0,
    #     alibi_slopes=None,
    #     return_softmax=return_softmax and dropout_p > 0,
    # )
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None, # out
        None, #alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        -1, #window_size[0],
        0, #window_size[1],
        return_softmax and dropout_p > 0,
        None,
    )
    return out, softmax_lse, rng_state

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@flash_attn.register_fake
def flash_attn_abstract(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        deterministic,
        return_softmax):
    out = q.new_empty(*q.shape)
    B, M, H, K = q.shape
    lse_shape = [B, H, M]
    softmax_lse = torch.empty(lse_shape, device=q.device, dtype=torch.float32)
    rng_state = torch.empty([2], device=q.device, dtype=torch.int64)
    return out, softmax_lse, rng_state

@torch.library.custom_op("wrapped_ops::flash_attn_bwd", mutates_args=())
def flash_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
    rng_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p,
        softmax_scale,
        causal,
        (-1, 0), #window_size,
        # 0.0, #softcap
        None, #alibi_slopes,
        deterministic,
        rng_state=rng_state,
    )
    dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
    dk = dk[..., : dout.shape[-1]]
    dv = dv[..., : dout.shape[-1]]
    return dq, dk, dv


@flash_attn_bwd.register_fake
def flash_attn_bwd_abstract(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dropout_p,
    softmax_scale,
    causal,
    deterministic,
    rng_state=None,):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

def _backward_flash_attn(ctx, dout: torch.Tensor, dsoftmax_lse: torch.Tensor, d_rng_states: torch.Tensor) -> torch.Tensor:
    q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
    dq, dk, dv = flash_attn_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        ctx.dropout_p,
        ctx.softmax_scale,
        ctx.causal,
        ctx.deterministic,
        rng_state=rng_state,
    )
    return dq, dk, dv, None, None, None, None, None, None, None

def setup_context(ctx, inputs, output):
    (q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal,
    deterministic,
    return_softmax) = inputs
    
    (out, softmax_lse, rng_state) = output

    ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
    ctx.dropout_p = dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.deterministic = deterministic

flash_attn.register_autograd(_backward_flash_attn, setup_context=setup_context)


class Attention(nn.Module):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        k_channels: int = None,
        v_channels: int = None,
    ):
        super().__init__()
        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = 'sbhd'
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale
        assert k_channels is None
        assert v_channels is None

        # assert attn_mask_type==AttnMaskType.causal


    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        packed_seq_params = None,
    ):
        # query = GradClip.apply(query)
        # key = GradClip.apply(key)
        # value = GradClip.apply(value)
        
        # torch.cuda.nvtx.range_push("Attention")
        query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
        out, _, _ =  flash_attn(
            query,
            key,
            value,
            0.0 if self.attention_dropout is None else self.attention_dropout,
            query.shape[-1] ** (-0.5) if self.softmax_scale is None else self.softmax_scale,
            causal=True,
            deterministic=False,
            return_softmax=True
        )
        out = out.transpose(0,1)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        assert len(out.shape)==3, f"{out.shape=}"
        # torch.cuda.nvtx.range_pop()
        return out


def flash_attn_inplace(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                dropout_p: float,
                softmax_scale: float,
                causal: bool,
                deterministic: bool,
                return_softmax: bool,
                out: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # out = None
    softmax_lse = None
    S_dmask = None 
    out_padded = None
    rng_state = None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(
        q,
        k,
        v,
        out,
        None, #alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        -1, #window_size[0],
        0, #window_size[1],
        return_softmax and dropout_p > 0,
        None,
    )
    return out, softmax_lse, rng_state