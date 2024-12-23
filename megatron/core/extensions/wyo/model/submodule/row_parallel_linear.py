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
from megatron.core.extensions.wyo.grad_clip import GradClip
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
    reduce_scatter_along_first_dim_in_tp_group,
    all_reduce_in_tp_group
)
from megatron.core.extensions.wyo.model.operator.layer_norm import layer_norm, layer_norm_bwd


class RowParallelLinear(TransformerEngineBaseModule):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__()
        self.config = config
        # assert skip_bias_add==False
        assert bias==False
        # assert config.sequence_parallel==True
        assert is_expert==False
        

        self.tp_size = get_tensor_model_parallel_world_size()

        self.out_features = output_size
        self.in_features = divide(input_size, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()
            
        self.sequence_parallel = (self.tp_size > 1) and config.sequence_parallel

        # prepare device and dtype
        device = torch.cuda.current_device()
        params_dtype = config.params_dtype

        
        # weight & bias for fc
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

    def forward(self, inp):
        # torch.cuda.nvtx.range_push("wyo RowParallelLinear")

        # inp = GradClip.apply(inp)

        # 1.fc
        fc_out = torch.matmul(inp, self.weight.t())

        if self.config.sequence_parallel:
            # 2.reduce-scatter
            out = reduce_scatter_along_first_dim_in_tp_group(fc_out)
        else:
            # 2.all-reduce
            out = all_reduce_in_tp_group(fc_out)

        # torch.cuda.nvtx.range_pop()

        return out, None