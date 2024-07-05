# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Some helper functions for implementing quantized modules"""
import copy
import inspect

from torch import nn

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR, QuantDescriptor
from .tensor_quantizer import TensorQuantizer


class QuantMixin:
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_output(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_output = copy.deepcopy(value)

    def init_quantizer(
        self, quant_desc_input, quant_desc_weight, num_layers=None, quant_desc_output=None
    ):
        """Helper function for __init__ of quantized module

        Create input and weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of
                :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
            quant_desc_weight: An instance of
                :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
            num_layers: An integer. Default None. If not None, create a list of quantizers.
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError(
                "{} should be only called by __init__ of quantized module.".format(__name__)
            )

        if quant_desc_output is None:
            quant_desc_output = QuantMixin.default_quant_desc_output

        self._fake_quant = True
        if (
            (not quant_desc_input.fake_quant)
            or (not quant_desc_weight.fake_quant)
            or (not quant_desc_output.fake_quant)
        ):
            raise ValueError("Only fake quantization is supported!")

        if num_layers is None:
            self._input_quantizer = TensorQuantizer(quant_desc_input)
            self._weight_quantizer = TensorQuantizer(quant_desc_weight)
            self._output_quantizer = TensorQuantizer(quant_desc_output)
            self._output_quantizer.disable()
        else:
            self._input_quantizers = nn.ModuleList(
                [TensorQuantizer(quant_desc_input) for _ in range(num_layers)]
            )
            self._weight_quantizers = nn.ModuleList(
                [TensorQuantizer(quant_desc_weight) for _ in range(num_layers)]
            )
            self._output_quantizers = nn.ModuleList(
                [TensorQuantizer(quant_desc_output) for _ in range(num_layers)]
            )
            for quantizer in self._output_quantizers:
                quantizer.disable()

    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer

    @property
    def output_quantizer(self):
        return self._output_quantizer


class QuantInputMixin:
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_output(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_output = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input, quant_desc_output=None):
        """Helper function for __init__ of simple quantized module

        Create input quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of
                :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError(
                "{} should be only called by __init__ of quantized module.".format(__name__)
            )

        if quant_desc_output is None:
            quant_desc_output = QuantMixin.default_quant_desc_output

        self._fake_quant = True
        if not quant_desc_input.fake_quant:
            raise ValueError("Only fake quantization is supported!")

        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._output_quantizer = TensorQuantizer(quant_desc_output)
        self._output_quantizer.disable()

    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def output_quantizer(self):
        return self._output_quantizer


# TODO: Remove this function
def pop_quant_desc_in_kwargs(quant_cls, input_only=False, **kwargs):
    """Pop quant descriptors in kwargs

    If there is no descriptor in kwargs, the default one in quant_cls will be used

    Arguments:
       quant_cls: A class that has default quantization descriptors
       input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

    Keyword Arguments:
       quant_desc_input: An instance of
            :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of input.
       quant_desc_weight: An instance of
            :class:`QuantDescriptor <ammo.torch.quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of weight.
    """
    quant_desc_input = kwargs.pop("quant_desc_input", quant_cls.default_quant_desc_input)
    if not input_only:
        quant_desc_weight = kwargs.pop("quant_desc_weight", quant_cls.default_quant_desc_weight)

    # Check if anything is left in **kwargs
    if kwargs:
        raise TypeError("Unused keys: {}".format(kwargs.keys()))

    if input_only:
        return quant_desc_input
    return quant_desc_input, quant_desc_weight
