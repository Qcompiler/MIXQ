# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module mapping for quantization."""
import torch.nn as nn

from . import nn as qnn

__all__ = ["QUANT_MODULE_MAPPING"]

QUANT_MODULE_MAPPING = {
    nn.Linear: qnn.QuantLinear.convert,
    nn.Conv1d: qnn.QuantConv1d.convert,
    nn.Conv2d: qnn.QuantConv2d.convert,
    nn.Conv3d: qnn.QuantConv3d.convert,
    nn.ConvTranspose1d: qnn.QuantConvTranspose1d.convert,
    nn.ConvTranspose2d: qnn.QuantConvTranspose2d.convert,
    nn.ConvTranspose3d: qnn.QuantConvTranspose3d.convert,
    nn.InstanceNorm1d: qnn.QuantInstanceNorm1d.convert,
    nn.InstanceNorm2d: qnn.QuantInstanceNorm2d.convert,
    nn.InstanceNorm3d: qnn.QuantInstanceNorm3d.convert,
    nn.MaxPool1d: qnn.QuantMaxPool1d.convert,
    nn.MaxPool2d: qnn.QuantMaxPool2d.convert,
    nn.MaxPool3d: qnn.QuantMaxPool3d.convert,
    nn.AvgPool1d: qnn.QuantAvgPool1d.convert,
    nn.AvgPool2d: qnn.QuantAvgPool2d.convert,
    nn.AvgPool3d: qnn.QuantAvgPool3d.convert,
    nn.AdaptiveAvgPool1d: qnn.QuantAdaptiveAvgPool1d.convert,
    nn.AdaptiveAvgPool2d: qnn.QuantAdaptiveAvgPool2d.convert,
    nn.AdaptiveAvgPool3d: qnn.QuantAdaptiveAvgPool3d.convert,
}
