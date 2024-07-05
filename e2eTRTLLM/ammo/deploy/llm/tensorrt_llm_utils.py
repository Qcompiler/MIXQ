# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils to convert model_config layers to tensorrt_llm modules."""

import tensorrt as trt
from tensorrt_llm.layers import Embedding, LayerNorm, RmsNorm
from tensorrt_llm.module import Module

from ammo.torch.export import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    EmbeddingConfig,
    LayernormConfig,
)


def get_tensor_parallel_group(tensor_parallel: int):
    """Returns the tensor_parallel_group config based on tensor_parallel."""
    return None if tensor_parallel == 1 else list(range(tensor_parallel))


def build_embedding_from_config(
    config: EmbeddingConfig,
    dtype: trt.DataType,
    rank=None,
    tensor_parallel: int = 1,
    instance_id: int = 0,
):
    """Returns the tensorrt_llm embedding layer from the embedding config."""
    # If the config is empty, return an empty impl.
    if config is None:
        return None
    trt_embedding = Embedding(
        config.local_vocab_size * tensor_parallel,
        config.weight.shape[1],
        dtype=dtype,
        tp_size=tensor_parallel,
        tp_group=get_tensor_parallel_group(tensor_parallel),
        tp_rank=rank,
        instance_id=instance_id,
    )
    trt_embedding.weight.value = config.weight
    return trt_embedding


def build_layernorm_from_config(config: LayernormConfig, dtype: trt.DataType):
    """Returns the tensorrt_llm layernorm layer from the torch layernorm."""
    # If the config is empty, return an empty impl.
    if config is None:
        return None

    if config.layernorm_type == LAYERNORM_DEFAULT:
        trt_layernorm = LayerNorm(
            normalized_shape=config.weight.shape[0], dtype=dtype, eps=config.eps
        )
        trt_layernorm.weight.value = config.weight
        if config.bias is not None:
            trt_layernorm.bias.value = config.bias
        # Bias of layernorm could be disabled for some models, e.g., MPT, filling zeros in this case.
        else:
            trt_layernorm.bias._value.fill(0)
    elif config.layernorm_type == LAYERNORM_RMS:
        trt_layernorm = RmsNorm(
            normalized_shape=config.weight.shape[0], dtype=dtype, eps=config.eps
        )
        trt_layernorm.weight.value = config.weight
    else:
        raise NotImplementedError(f"{config.layernorm_type} not supported")
    return trt_layernorm


def print_tensorrt_llm(name: str, tensorrt_llm_module: Module):
    """Prints the tensorrt llm structure including weights and related data for debugging purpose."""
    for tensor_name in [
        "weight",
        "bias",
        "activation_scaling_factor",
        "weights_scaling_factor",
        "prequant_scaling_factor",
        "kv_orig_quant_scale",
        "kv_quant_orig_scale",
    ]:
        if hasattr(tensorrt_llm_module, tensor_name):
            tensor = getattr(tensorrt_llm_module, tensor_name)
            if tensor is not None:
                print(
                    f"{name}.{tensor_name}:{tensor._value.dtype}:{tensor._value.shape}:\n{tensor._value}"
                )

    for k, v in tensorrt_llm_module.named_children():
        print_tensorrt_llm(f"{name}.{k}({v._get_name()})", v)
