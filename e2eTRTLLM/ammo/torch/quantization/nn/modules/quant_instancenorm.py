# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantized instance normalization module."""

from torch.nn.modules import instancenorm

from .quant_module import QuantInputBase

__all__ = ["QuantInstanceNorm1d", "QuantInstanceNorm2d", "QuantInstanceNorm3d"]


class QuantInstanceNorm1d(QuantInputBase, instancenorm.InstanceNorm1d):
    """Applies Quantized Instance Normalization over a 3D input."""

    pass


class QuantInstanceNorm2d(QuantInputBase, instancenorm.InstanceNorm2d):
    """Applies Quantized Instance Normalization over a 4D input."""

    pass


class QuantInstanceNorm3d(QuantInputBase, instancenorm.InstanceNorm3d):
    """Applies Quantized Instance Normalization over a 5D input."""

    pass
