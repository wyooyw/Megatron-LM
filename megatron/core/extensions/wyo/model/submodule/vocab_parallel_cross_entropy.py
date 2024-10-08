from typing import Tuple

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy, _VocabParallelCrossEntropy

@torch.library.custom_op("wrapped_ops::vocab_parallel_cross_entropy", mutates_args=())
def _vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits
    )
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
    )

    # Get the partition's vocab indices
    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

    (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
        VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )
    )

    # All reduce is needed to get the chunks from other GPUs.
    torch.distributed.all_reduce(
        predicted_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )

    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )

    vocab_size = exp_logits.size(-1)
    if label_smoothing > 0:
        """
        We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
        = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
        = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
        = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
        = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
        = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
        From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
        """
        assert 1.0 > label_smoothing > 0.0
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)

        # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
        log_probs = torch.log(exp_logits)
        mean_log_probs = log_probs.mean(dim=-1)
        loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

    # ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

    # # Store softmax, target-mask and masked-target for backward pass.
    # ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

    return loss, exp_logits, target_mask, masked_target_1d



@_vocab_parallel_cross_entropy.register_fake
def _vocab_parallel_cross_entropy_abstract(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlen, bs, vocab_size_per_tp_rank = vocab_parallel_logits.shape
    loss = torch.empty(
        (seqlen, bs), 
        dtype=torch.float32,
        device=vocab_parallel_logits.device
    )
    exp_logits = torch.empty(
        (seqlen, bs, vocab_size_per_tp_rank), 
        dtype=torch.float32,
        device=vocab_parallel_logits.device
    )
    target_mask = torch.empty(
        (seqlen, bs), 
        dtype=bool,
        device=vocab_parallel_logits.device
    )
    masked_target_1d = torch.empty(
        (seqlen * bs,), 
        dtype=torch.int64,
        device=vocab_parallel_logits.device
    )
    return loss, exp_logits, target_mask, masked_target_1d

@torch.library.custom_op("wrapped_ops::vocab_parallel_cross_entropy_backward", mutates_args=())
def vocab_parallel_cross_entropy_backward(
    grad_output: torch.Tensor,
    softmax: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
    label_smoothing: float,
    vocab_size: int
) -> torch.Tensor:

    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )

    if label_smoothing > 0:
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)
        grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
        average_grad = 1 / vocab_size
        grad_2d[arange_1d, :] -= smoothing * average_grad

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
    else:
        grad_input = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
        )

    return grad_input.clone()


@vocab_parallel_cross_entropy_backward.register_fake
def vocab_parallel_cross_entropy_backward_abstract(
    grad_output: torch.Tensor,
    softmax: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
    label_smoothing: float,
    vocab_size: int
) -> torch.Tensor:
    # vocab_size is vocab_size_per_tp_rank
    seqlen, bs = grad_output.shape
    grad_input = torch.empty(
        (seqlen, bs, vocab_size), 
        dtype=grad_output.dtype,
        device=grad_output.device
    )
    return grad_input

def _backward_vocab_parallel_cross_entropy(ctx,
        d_loss, d_exp_logits, d_target_mask, d_masked_target_1d):
    
    exp_logits, target_mask, masked_target_1d = ctx.saved_tensors

    d_input = vocab_parallel_cross_entropy_backward(
        d_loss,
        exp_logits,
        target_mask,
        masked_target_1d,
        ctx.label_smoothing,
        ctx.vocab_size
    )
    
    return d_input, None, None

def setup_context(ctx, inputs, output):
    (
        vocab_parallel_logits, 
        target, 
        label_smoothing
    ) = inputs

    (
        loss, 
        exp_logits, 
        target_mask, 
        masked_target_1d
    ) = output

    vocab_size = exp_logits.size(-1)
    ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

    # Store softmax, target-mask and masked-target for backward pass.
    ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

_vocab_parallel_cross_entropy.register_autograd(_backward_vocab_parallel_cross_entropy, setup_context=setup_context)


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0
) :
    loss, exp_logits, target_mask, masked_target_1d = _vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing)
    return loss