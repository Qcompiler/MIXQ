# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Base class for quantization modules."""
import copy
import types

import torch.nn as nn

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR, QuantDescriptor
from .tensor_quantizer import TensorQuantizer


class _QuantModuleBase:
    @classmethod
    def quantized_forward(cls, self, input, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def convert(cls, module):
        assert not hasattr(
            module, "_original_forward"
        ), "module already has an _original_forward attribute! module forward cannot be patched."
        module._original_forward = module.forward
        module.forward = types.MethodType(cls.quantized_forward, module)
        return module


class QuantLinearConvBase(_QuantModuleBase, nn.Module):
    """Base class for quantized linear modules.

    Quantized linear modules are modules where both the input and the weight are quantized.
    This class also provides a classmethod to convert a linear module  such as `nn.ConvNd`,
    `nn.Linear` in-place to a quantized one.
    """

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, quant_desc_weight=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        super().__init__(*args, **kwargs)
        self.convert(self, quant_desc_input, quant_desc_weight)

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """Set the class default input quantization descriptor (legacy method)."""
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """Set the class default weight quantization descriptor (legacy method)."""
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    @classmethod
    def quantized_forward(cls, self, input, *args, **kwargs):
        """Quantize the input and weight before calling the original forward method."""
        self.__dict__["weight"] = self.weight_quantizer(self.weight)
        output = self._original_forward(self.input_quantizer(input), *args, **kwargs)
        del self.__dict__["weight"]
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    @classmethod
    def convert(cls, module, quant_desc_input=None, quant_desc_weight=None):
        """Patch the module's forward method to quantize the input and weight."""
        module = super().convert(module)

        module.input_quantizer = TensorQuantizer(quant_desc_input or cls.default_quant_desc_input)
        module.weight_quantizer = TensorQuantizer(
            quant_desc_weight or cls.default_quant_desc_weight
        )
        module.output_quantizer = TensorQuantizer(cls.default_quant_desc_output)
        module.output_quantizer.disable()

        return module


class QuantInputBase(_QuantModuleBase, nn.Module):
    """Base class for modules where only the input is quantized.

    This class also provides a classmethod to convert any nn.Module where the input is a tensor to a quantized one.
    """

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        super().__init__(*args, **kwargs)
        self.convert(self, quant_desc_input)

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """Set the class default input quantization descriptor (legacy method)."""
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def quantized_forward(cls, self, input, *args, **kwargs):
        """Quantize the input before calling the original forward method."""
        output = self._original_forward(self.input_quantizer(input), *args, **kwargs)
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    @classmethod
    def convert(cls, module, quant_desc_input=None):
        """Patch the module's forward method to quantize the input."""
        module = super().convert(module)

        module.input_quantizer = TensorQuantizer(quant_desc_input or cls.default_quant_desc_input)
        module.output_quantizer = TensorQuantizer(cls.default_quant_desc_output)
        module.output_quantizer.disable()

        return module
