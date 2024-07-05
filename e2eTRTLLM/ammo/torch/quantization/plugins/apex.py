# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for apex linear layers."""

import apex.transformer.tensor_parallel.layers as apex_parallel

from ..module_mapping import QUANT_MODULE_MAPPING
from ..nn import QuantLinear

_functionals_to_replace = [(apex_parallel, "linear_with_grad_accumulation_and_async_allreduce")]


def convertQuantParallelLinear(linear):  # noqa
    """Convert apex parallel linear layer to quantized parallel linear layer."""
    assert type(linear).__name__ in ["ColumnParallelLinear", "RowParallelLinear"]
    return QuantLinear.insert_quantizers_and_replace_fns(linear, _functionals_to_replace)


QUANT_MODULE_MAPPING[apex_parallel.ColumnParallelLinear] = convertQuantParallelLinear
QUANT_MODULE_MAPPING[apex_parallel.RowParallelLinear] = convertQuantParallelLinear
