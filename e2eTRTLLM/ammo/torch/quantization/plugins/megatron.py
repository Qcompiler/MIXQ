# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for megatron linear layers."""


import megatron.core.tensor_parallel.layers as megatron_parallel

from ..module_mapping import QUANT_MODULE_MAPPING
from ..nn import QuantLinear

__all__ = []

_functionals_to_replace = [
    (megatron_parallel, "linear_with_grad_accumulation_and_async_allreduce"),
    (megatron_parallel, "linear_with_frozen_weight"),
]


def convertQuantParallelLinear(linear):  # noqa
    """Convert megatron parallel linear layer to quantized parallel linear layer."""
    assert type(linear).__name__ in ["ColumnParallelLinear", "RowParallelLinear"]
    return QuantLinear.insert_quantizers_and_replace_fns(linear, _functionals_to_replace)


QUANT_MODULE_MAPPING[megatron_parallel.ColumnParallelLinear] = convertQuantParallelLinear
QUANT_MODULE_MAPPING[megatron_parallel.RowParallelLinear] = convertQuantParallelLinear
