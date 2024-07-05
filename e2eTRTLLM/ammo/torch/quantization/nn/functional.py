# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Some supportive functions."""
import warnings

import torch
from torch.autograd import Function


class ClipFunction(Function):
    """An universal tensor clip function.

    Pytorch's clamp() only supports scalar range and doesn't support broadcast. This implementation uses min/max which
    is more genaral. The gradient is defined according to IBM's PACT paper https://arxiv.org/abs/1805.06085, which is
    also the behavior of Tensorflow's clip_by_value()
    """

    @staticmethod
    def forward(ctx, input, clip_value_min, clip_value_max):
        """Forward pass for the clip function."""
        output = torch.min(input, clip_value_max)
        output = torch.max(output, clip_value_min)
        ctx.save_for_backward(input, clip_value_min, clip_value_max)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for the clip function."""
        input, clip_value_min, clip_value_max = ctx.saved_tensors
        min_mask = (input > clip_value_min).to(grad_output.dtype)
        max_mask = (input < clip_value_max).to(grad_output.dtype)
        grad_input = grad_output * min_mask * max_mask

        if clip_value_min.requires_grad or clip_value_max.requires_grad:
            warnings.warn("Learning enabled for clip min/max. This is an experimental feature.")
        if clip_value_min.numel() != 1 or clip_value_max.numel() != 1:
            raise ValueError(
                "Learnable min/max can only be scalar, got size %s and %s."
                % (clip_value_min.size(), clip_value_max.size())
            )

        # Ensure the dtypes of min/max grads matches the input dtype
        # This might be necessary if running w/ AMP which will cast to fp32 before `sum()`
        grad_clip_value_min = (
            (grad_output * (1.0 - min_mask)).sum().to(clip_value_min.dtype)
            if clip_value_min.requires_grad
            else None
        )
        grad_clip_value_max = (
            (grad_output * (1.0 - max_mask)).sum().to(clip_value_min.dtype)
            if clip_value_max.requires_grad
            else None
        )

        return grad_input, grad_clip_value_min, grad_clip_value_max


clip = ClipFunction.apply
