# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantized Linear."""
import types
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Any, List, Optional, Tuple

import torch.nn as nn

from ammo.torch.quantization import tensor_quant
from ammo.torch.quantization.nn.modules.quant_module import QuantLinearConvBase
from ammo.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from ammo.torch.quantization.utils import replace_function

__all__ = ["Linear", "QuantLinear"]


@contextmanager
def _multi_context(*cms):
    """Context manager enabling variable number of context managers."""
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


class QuantLinear(QuantLinearConvBase, nn.Linear):
    """Quantized version of nn.Linear."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    _functionals_to_replace = [(nn.functional, "linear")]

    @classmethod
    def convert(cls, module, quant_desc_input=None, quant_desc_weight=None):
        """Convert a nn.Linear module to a quantized one."""
        return cls.insert_quantizers_and_replace_fns(
            module, QuantLinear._functionals_to_replace, quant_desc_input, quant_desc_weight
        )

    @staticmethod
    def insert_quantizers_and_replace_fns(
        linear: nn.Module,
        functionals_to_replace: List[Tuple[Any, str]],
        default_quant_desc_input: Optional[tensor_quant.QuantDescriptor] = None,
        default_quant_desc_weight: Optional[tensor_quant.QuantDescriptor] = None,
    ):
        """Convert linear layer to quantized version.

        Conversion is done by inserting quantizers and replacing functionals with quantized counterparts.
        Replacing functionals is better than monkey patching the linear module forward method to support
        linear modules from megatron/apex and linear modules with huggingface accelerate hooks.

        Args:
            linear: Linear layer to be converted
            functionals_to_replace: A list of tuple (package, function_name) where function_name
                specifies the function that needs to be replaced from the corresponding package.
                For example, [(torch.nn.functional, "linear")] will replace torch.nn.functional.linear
                with its quantized version.
            default_quant_desc_input: Default quantization descriptor for input
            default_quant_desc_weight: Default quantization descriptor for weight

        """
        assert not hasattr(
            linear, "_original_forward"
        ), "module already has an _original_forward attribute! module forward cannot be patched."

        # initialize input quantizer and weight quantizer
        linear.input_quantizer = TensorQuantizer(
            default_quant_desc_input or QuantLinear.default_quant_desc_input
        )
        linear.weight_quantizer = TensorQuantizer(
            default_quant_desc_weight or QuantLinear.default_quant_desc_weight
        )
        linear.output_quantizer = TensorQuantizer(
            default_quant_desc_input or QuantLinear.default_quant_desc_input
        )
        linear.output_quantizer.disable()

        for package, fn_name in functionals_to_replace:
            if hasattr(package, fn_name):
                quantized_fn = partial(
                    QuantLinear.quantized_linear_fn, package, "_" + fn_name, linear
                )
                setattr(linear, "_quantized_" + fn_name, quantized_fn)

                # A fix for megatron method `linear_with_grad_accumulation_and_async_allreduce`
                # which has a  method attribute "warned" as in
                # https://github.com/NVIDIA/Megatron-LM/blob/fab0bd693ec5be55b32c4f12a1ea44766ec63448/megatron/core/tensor_parallel/layers.py#L539
                if hasattr(getattr(package, fn_name), "__dict__"):
                    quantized_fn.__dict__.update(getattr(package, fn_name).__dict__)

        def new_forward(self, *args, **kwargs):
            with _multi_context(
                *(
                    replace_function(package, fn_name, getattr(self, "_quantized_" + fn_name))
                    for package, fn_name in functionals_to_replace
                    if hasattr(package, fn_name)
                )
            ):
                return self._original_forward(*args, **kwargs)

        linear._original_forward = linear.forward
        linear.forward = types.MethodType(new_forward, linear)

        return linear

    @staticmethod
    def quantized_linear_fn(package, func_name, self, input, weight, *args, **kwargs):
        """Quantized version of a generic linear functional."""
        output = getattr(package, func_name)(
            self.input_quantizer(input),
            self.weight_quantizer(weight),
            *args,
            **kwargs,
        )
        return self.output_quantizer(output)


Linear = QuantLinear
