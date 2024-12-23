from megatron.core.transformer.enums import AttnMaskType

from megatron.core.transformer.attention import SelfAttention as megatron_SelfAttention
import torch
from megatron.core.extensions.wyo.grad_clip import GradClip


class SelfAttention(megatron_SelfAttention):
    def __init__(
        self,
        config,
        submodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(config, submodules, layer_number, attn_mask_type)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # if SplitAlongDim is not None:

        #     # [sq, b, ng, (np/ng + 2) * hn]
        #     # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        #     (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        # else:

        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value