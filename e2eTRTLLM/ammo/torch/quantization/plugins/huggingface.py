# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for huggingface layers."""
import types

import torch.nn as nn
import transformers

from .. import tensor_quant
from ..module_mapping import QUANT_MODULE_MAPPING
from ..nn import QuantLinear
from ..nn.modules.quant_module import QuantLinearConvBase

__all__ = []


# transformers.modeling_utils.Conv1D used in HF-GPT2 is not a real Conv1D
# It is actually a Linear layer where weight is transposed and torch.addmm is used
def convert_conv1d(conv):
    """Convert hf Conv1d layer to quantized Conv1d layer."""
    conv.weight = nn.Parameter(conv.weight.T)
    conv.forward = types.MethodType(nn.Linear.forward, conv)
    return QuantLinear.convert(conv)


QUANT_MODULE_MAPPING[transformers.modeling_utils.Conv1D] = convert_conv1d
if hasattr(transformers.models, "falcon") and hasattr(
    transformers.models.falcon.modeling_falcon, "FalconLinear"
):

    class QuantFalconLinear(
        transformers.models.falcon.modeling_falcon.FalconLinear, QuantLinearConvBase
    ):
        default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    QUANT_MODULE_MAPPING[transformers.models.falcon.modeling_falcon.FalconLinear] = (
        QuantFalconLinear.convert
    )


def register_falcon_linears_on_the_fly(model):
    """Register Falcon linear modules as a QUANT_MODULE.

    Certain falcon models (for example, falcon 40b) use remote code, which are loaded dynamically, to build their model.
    Therefore, we need to register the linear on the fly before quantization.
    """
    if type(model).__name__ in ["RWForCausalLM", "FalconForCausalLM"]:
        from ammo.torch.quantization import tensor_quant
        from ammo.torch.quantization.nn.modules.quant_module import QuantLinearConvBase

        linear_type = type(model.transformer.h[0].self_attention.dense)

        class QuantFalconLinearRW1B(linear_type, QuantLinearConvBase):  # type: ignore
            default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

        QUANT_MODULE_MAPPING[linear_type] = QuantFalconLinearRW1B.convert
