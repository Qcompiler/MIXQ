# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/operators/conv.py
# and https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/operators/gather.py
#
# MIT License
#
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Additional or modified QDQ operators on top ORT quantized operators."""
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase

from ..op_types import is_normalization_op


class QDQNormalization(QDQOperatorBase):
    """By default, ORT does not quantize Normalization ops. This module is intended to help with that.

    Note. QDQOperatorBase is not sufficient for dynamic input only quantization.
    """

    def __init__(self, onnx_quantizer, onnx_node):
        """Normalization quantizer init."""
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Main function to quantize the Normalization ops."""
        node = self.node
        assert is_normalization_op(node.op_type)

        # Quantize only the dynamic input (first input to the op)
        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])


class QDQConvTranspose(QDQOperatorBase):
    """QDQ for ConvTranspose operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """ConvTranspose quantizer init."""
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Main function to quantize the ConvTranspose ops."""
        node = self.node
        assert node.op_type == "ConvTranspose"

        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        if self.quantizer.is_per_channel():
            channel_axis = self.quantizer.qdq_op_type_per_channel_support_to_axis.get(
                node.op_type, 0
            )
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], channel_axis)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

        if len(node.input) == 3:
            self.quantizer.quantize_bias_tensor(node.input[2], node.input[0], node.input[1])
