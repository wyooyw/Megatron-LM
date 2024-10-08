from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
import torch
import torch.nn as nn
import importlib
from typing import Tuple, List
from typing import Callable
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_te_version, is_te_min_version

from transformer_engine.pytorch.utils import (
    divide,
    get_default_init_method,
    init_method_constant,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    requires_grad,
)
from transformer_engine.pytorch.module.base import (
    TransformerEngineBaseModule
)
from megatron.core.extensions.wyo.model.communicate.communicate import (
    gather_along_first_dim_in_tp_group,
    reduce_scatter_along_first_dim_in_tp_group
)
from megatron.core.extensions.wyo.model.operator.layer_norm import layer_norm, layer_norm_bwd

# @torch.library.custom_op("wrapped_ops::layernorm_column_parallel_linear", mutates_args=())
# def layernorm_column_parallel_linear_sp(
#     inp: torch.Tensor,
#     fc_weight: torch.Tensor,
#     # fc_bias: torch.Tensor,

#     ln_weight: torch.Tensor,
#     ln_bias: torch.Tensor,
#     ln_normalized_shape: int,
#     ln_eps: float,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

#     # step 1: layernorm
#     ln_out, ln_mean, ln_invvar = layer_norm(
#         inp,
#         ln_weight,
#         ln_bias,
#         normalized_shape=ln_normalized_shape,
#         eps=ln_eps,
#         memory_efficient=True
#     )

#     # step 2: all-gather
#     ag_out = gather_along_first_dim_in_tp_group(ln_out)

#     # step 3: gemm
#     fc_out = torch.matmul(ag_out, fc_weight.t())
    
#     return fc_out, ln_out, ln_mean, ln_invvar

# @layernorm_column_parallel_linear_sp.register_fake
# def layernorm_column_parallel_linear_sp_abstract(
#     input: torch.Tensor,
#     fc_weight: torch.Tensor,
#     # fc_bias: torch.Tensor,

#     ln_weight: torch.Tensor,
#     ln_bias: torch.Tensor,
#     ln_normalized_shape: int,
#     ln_eps: float,
# ):
#     out_feature, in_feature = fc_weight.shape
#     tp_size = get_tensor_model_parallel_world_size()

#     output_shape = list(input.size())
#     output_shape[0] *= tp_size
#     output_shape[-1] = out_feature
#     output = torch.empty(
#         output_shape,
#         dtype=input.dtype,
#         device=input.device,
#         memory_format=torch.contiguous_format,
#     )

#     ln_out = torch.empty(
#         list(input.size()),
#         dtype=input.dtype,
#         device=input.device,
#         memory_format=torch.contiguous_format,
#     )

#     # TODO: force fp32?
#     mean_dim = len(input.shape) - 1
#     mean = input.new_empty(*input.shape[:mean_dim])
#     invvar = input.new_empty(*input.shape[:mean_dim])
    
#     return output, ln_out, mean, invvar

# def _backward_layernorm_column_parallel_linear_sp(ctx, d_fc_out, _d_ln_out, _d_ln_mean, _d_ln_invvar):
#     """
#     This function will generate at least 5 ops in fx.graph:
#         allgather
#         matmul
#         matmul
#         reduce_scatter
#         layernorm_bwd
#     """
    
#     fc_weight, ln_out, ln_weight, ln_bias, ln_mean, ln_invvar = ctx.saved_tensors

#     # re-all-gather 
#     inp_total = gather_along_first_dim_in_tp_group(ln_out)

#     # backward of fc
#     # dI = dO * W
#     d_fc_inp = torch.matmul(d_fc_out, fc_weight)
#     # dW = dO^T * I
#     d_fc_out = d_fc_out.reshape(-1, d_fc_out.shape[-1])
#     inp_total = inp_total.reshape(-1, inp_total.shape[-1])
#     d_fc_weight = torch.matmul(d_fc_out.t(), inp_total)

#     # backward of all-gather(reduce-scatter)
#     d_ln_out = reduce_scatter_along_first_dim_in_tp_group(d_fc_inp)

#     # backward of layernorm
#     d_ln_input, d_ln_weight, d_ln_bias = layer_norm_bwd(
#         d_ln_out,
#         ln_out,
#         ln_weight,
#         ln_bias,
#         ln_mean,
#         ln_invvar,
#         ctx.ln_normalized_shape,
#         ctx.ln_eps,
#         True # ctx.ln_memory_efficient
#     )
    
#     return d_ln_input, d_fc_weight, d_ln_weight, d_ln_bias, None, None

# def setup_context(ctx, inputs, output):
#     (
#         input,
#         fc_weight,

#         ln_weight,
#         ln_bias,
#         ln_normalized_shape,
#         ln_eps,
#     ) = inputs
    
#     (fc_output, ln_out, ln_mean, ln_invvar) = output

#     ctx.ln_normalized_shape = ln_normalized_shape
#     ctx.ln_eps = ln_eps
    
#     ctx.save_for_backward(fc_weight, ln_out, ln_weight, ln_bias, None, ln_invvar)
    

# layernorm_column_parallel_linear_sp.register_autograd(_backward_layernorm_column_parallel_linear_sp, setup_context=setup_context)

class LayernormColumnParallelLinearSp(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        fc_weight: torch.Tensor,
        # fc_bias: torch.Tensor,

        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        ln_normalized_shape: int,
        ln_eps: float
    ):
        # step 1: layernorm
        ln_out, ln_mean, ln_invvar = layer_norm(
            inp,
            ln_weight,
            ln_bias,
            normalized_shape=ln_normalized_shape,
            eps=ln_eps,
            memory_efficient=True
        )

        # step 2: all-gather
        ag_out = gather_along_first_dim_in_tp_group(ln_out)

        # step 3: gemm
        fc_out = torch.matmul(ag_out, fc_weight.t())

        ctx.ln_normalized_shape = ln_normalized_shape
        ctx.ln_eps = ln_eps
        ctx.save_for_backward(fc_weight, ln_out, ln_weight, ln_bias, ln_mean, ln_invvar)

        return fc_out
    
    @staticmethod
    def backward(
        ctx, d_fc_out
    ):
        fc_weight, ln_out, ln_weight, ln_bias, ln_mean, ln_invvar = ctx.saved_tensors

        # re-all-gather 
        inp_total = gather_along_first_dim_in_tp_group(ln_out)

        # backward of fc
        # dI = dO * W
        d_fc_inp = torch.matmul(d_fc_out, fc_weight)
        # dW = dO^T * I
        d_fc_out = d_fc_out.reshape(-1, d_fc_out.shape[-1])
        inp_total = inp_total.reshape(-1, inp_total.shape[-1])
        d_fc_weight = torch.matmul(d_fc_out.t(), inp_total)

        # backward of all-gather(reduce-scatter)
        d_ln_out = reduce_scatter_along_first_dim_in_tp_group(d_fc_inp)

        # backward of layernorm
        d_ln_input, d_ln_weight, d_ln_bias = layer_norm_bwd(
            d_ln_out,
            ln_out,
            ln_weight,
            ln_bias,
            ln_mean,
            ln_invvar,
            ctx.ln_normalized_shape,
            ctx.ln_eps,
            True # ctx.ln_memory_efficient
        )
        
        return d_ln_input, d_fc_weight, d_ln_weight, d_ln_bias, None, None

class LayerNormColumnParallelLinear(TransformerEngineBaseModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__()
        self.config = config
        # assert skip_bias_add==False
        assert bias==False
        assert config.sequence_parallel==True
        assert gather_output==False
        assert is_expert==False
        

        self.tp_size = get_tensor_model_parallel_world_size()

        self.out_features = divide(output_size, self.tp_size)
        self.in_features = input_size

        if init_method is None:
            init_method = get_default_init_method()
            
        self.sequence_parallel = (self.tp_size > 1) and config.sequence_parallel

        # prepare device and dtype
        device = torch.cuda.current_device()
        params_dtype = config.params_dtype

        # Prepare layernorm
        self.eps = self.config.layernorm_epsilon
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(self.in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.config.layernorm_zero_centered_gamma)),
        )
        if self.config.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(self.in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

        # weightr & bias for fc
        fc_weight_name = "weight"
        fc_weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        self.register_parameter(
            fc_weight_name,
            torch.nn.Parameter(fc_weight_tensor),
            init_fn=init_method
        )
        
        fc_bias_name = "bias"
        fc_bias_tensor = None
        if bias:
            fc_bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )
            self.register_parameter(
                fc_bias_name,
                torch.nn.Parameter(fc_bias_tensor),
                init_fn=init_method
            )
            

        self.reset_parameters(defer_init=(device == "meta"))
    
    def forward(self, x):
        # torch.cuda.nvtx.range_push("wyo LayerNormColumnParallelLinear")
        # out, _, _, _ = layernorm_column_parallel_linear_sp(
        #         x,
        #         fc_weight=self.weight,

        #         ln_weight=self.layer_norm_weight,
        #         ln_bias=self.layer_norm_bias,
        #         ln_normalized_shape=x.shape[-1],
        #         ln_eps=self.config.layernorm_epsilon ,
        # )
        out = LayernormColumnParallelLinearSp.apply(
            x,
            self.weight,

            self.layer_norm_weight,
            self.layer_norm_bias,
            x.shape[-1],
            self.config.layernorm_epsilon
        )
        # torch.cuda.nvtx.range_pop()
        output_bias = None
        return out, output_bias

