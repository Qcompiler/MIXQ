# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Implement a clip module as pytorch only has a simple clamp function."""
import torch
from torch import nn
from torch.nn.parameter import Parameter

from .. import functional as QF

__all__ = ["Clip"]


class Clip(nn.Module):
    """Clip tensor.

    Args:
        clip_value_min: A number or tensor of lower bound to clip
        clip_value_max: A number of tensor of upper bound to clip
        learn_min: A boolean. If True, learn min. clip_value_min will be used to initialize. Default False
        learn_max: A boolean. Similar as learn_min but for max.

    Raises:
        ValueError:
    """

    def __init__(self, clip_value_min, clip_value_max, learn_min=False, learn_max=False):
        """Initialize."""
        super(Clip, self).__init__()
        if learn_min:
            if not isinstance(clip_value_min, float) and clip_value_min.size != 1:
                raise ValueError(
                    "clip_value_min/clip_value_max must be scalar for initilizing learnable range."
                )
            self.clip_value_min = Parameter(torch.tensor(clip_value_min))
        else:
            self.clip_value_min = clip_value_min

        if learn_max:
            if not isinstance(clip_value_max, float) and clip_value_max.size != 1:
                raise ValueError(
                    "clip_value_min/clip_value_max must be scalar for initilizing learnable range."
                )
            self.clip_value_max = Parameter(torch.tensor(clip_value_max))
        else:
            self.clip_value_max = clip_value_max

    def forward(self, inputs):
        """Clip input tensor."""
        outputs = QF.clip(inputs, self.clip_value_min, self.clip_value_max)
        return outputs
