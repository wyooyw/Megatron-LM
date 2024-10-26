from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
import torch
import torch.nn as nn
import importlib
from typing import Tuple, List
from typing import Callable
from typing import Any, Callable, List, Optional, Tuple
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
from megatron.core.tensor_parallel.layers import ColumnParallelLinear as megatron_ColumnParallelLinear

class ColumnParallelLinear(megatron_ColumnParallelLinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,  # Not used
            disable_grad_reduce=disable_grad_reduce,
        )

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )
            
        # torch.cuda.nvtx.range_push("wyo RowParallelLinear")

        if self.config.sequence_parallel:
            # 1.all-gather
            ag_out = gather_along_first_dim_in_tp_group(input_)
        else:
            ag_out = input_

        # 2.fc
        fc_out = torch.matmul(ag_out, weight.t())

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_along_first_dim_in_tp_group(fc_out)
        else:
            output = fc_out

        return output, None