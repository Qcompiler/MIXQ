# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Customization for Nemo Megatron GPT."""

import types

import torch

# New nemo version should depend on megatron.core
from nemo.collections.nlp.modules.common.megatron.attention import CoreAttention

from ..module_mapping import QUANT_MODULE_MAPPING
from ..nn import QuantLinear, TensorQuantizer
from ..utils import replace_function

__all__ = []


def convertCoreAttention(attention):  # noqa
    """Current implementation uses context manager to replace torch functional with quantized version.

    This is a hacky way to do it, but it works for now.
    In the future if BMM quantization becomes the standard, we should work with Nemo
    team to replace bmm functional with bmm modules.
    """
    assert not hasattr(
        attention, "_original_forward"
    ), "module already has an _original_forward attribute! module forward cannot be patched."

    def new_forward(self, *args, **kwargs):
        with replace_function(torch, "bmm", self.quantized_bmm), replace_function(
            torch, "baddbmm", self.quantized_baddbmm
        ):
            return self._original_forward(*args, **kwargs)

    attention._original_forward = attention.forward
    attention.forward = types.MethodType(new_forward, attention)
    attention.quantized_bmm = types.MethodType(quantized_bmm, attention)
    attention.quantized_baddbmm = types.MethodType(quantized_baddbmm, attention)

    attention.q_bmm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
    attention.k_bmm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
    attention.v_bmm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)

    return attention


def quantized_bmm(self, input, mat2, *args, **kwargs):
    """Quantized version of BMM2 in nemo CoreAttention."""
    attn, v = input, mat2
    return torch._bmm(attn, self.v_bmm_quantizer(v), *args, **kwargs)


def quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    """Quantized version of BMM1 in nemo CoreAttention."""
    q, k = batch1, batch2
    return torch._baddbmm(input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs)


QUANT_MODULE_MAPPING[CoreAttention] = convertCoreAttention
