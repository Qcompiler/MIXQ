# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantized convolution."""
import torch.nn as nn

from ammo.torch.quantization import tensor_quant
from ammo.torch.quantization.nn.modules.quant_module import QuantLinearConvBase

__all__ = [
    "Conv2d",
    "QuantConv2d",
    "Conv3d",
    "QuantConv3d",
    "Conv1d",
    "QuantConv1d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "QuantConvTranspose1d",
    "QuantConvTranspose2d",
    "QuantConvTranspose3d",
]


class QuantConv1d(QuantLinearConvBase, nn.Conv1d):
    """Quantized 1D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL


class QuantConv2d(QuantLinearConvBase, nn.Conv2d):
    """Quantized 2D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL


class QuantConv3d(QuantLinearConvBase, nn.Conv3d):
    """Quantized 3D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL


class QuantConvTranspose1d(QuantLinearConvBase, nn.ConvTranspose1d):
    """Quantized 1D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL


class QuantConvTranspose2d(QuantLinearConvBase, nn.ConvTranspose2d):
    """Quantized 2D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL


class QuantConvTranspose3d(QuantLinearConvBase, nn.ConvTranspose3d):
    """Quantized 3D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL


# Define alias with Quant prefix
Conv1d = QuantConv1d
Conv2d = QuantConv2d
Conv3d = QuantConv3d
ConvTranspose1d = QuantConvTranspose1d
ConvTranspose2d = QuantConvTranspose2d
ConvTranspose3d = QuantConvTranspose3d
