import torch
from torch.autograd import Function

class GradClip(Function):
    @staticmethod
    def forward(ctx, input, clip_value):
        ctx.clip_value = clip_value
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -ctx.clip_value, ctx.clip_value)
        return grad_input, None