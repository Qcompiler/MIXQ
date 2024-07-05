# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The utils to convert a tensorrt_llm network to a quantized network."""

from typing import Optional, Union

import torch
from tensorrt_llm.layers import Linear, RowLinear
from tensorrt_llm.quantization.layers import (
    FP8Linear,
    FP8RowLinear,
    Int8SmoothQuantLinear,
    Int8SmoothQuantRowLinear,
    WeightOnlyGroupwiseQuantColumnLinear,
    WeightOnlyGroupwiseQuantRowLinear,
)

from ammo.torch.export import (
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_NONE,
    LinearConfig,
    ModelConfig,
    QKVConfig,
)

linear_qantized_trtllm_layer = {
    QUANTIZATION_FP8: FP8Linear,
    QUANTIZATION_INT8_SQ: Int8SmoothQuantLinear,
    QUANTIZATION_INT4_AWQ: WeightOnlyGroupwiseQuantColumnLinear,
}

row_linear_qantized_trtllm_layer = {
    QUANTIZATION_FP8: FP8RowLinear,
    QUANTIZATION_INT8_SQ: Int8SmoothQuantRowLinear,
    QUANTIZATION_INT4_AWQ: WeightOnlyGroupwiseQuantRowLinear,
}


def _should_skip_quantization(quantization, weights_scaling_factor, activation_scaling_factor):
    if quantization == QUANTIZATION_INT4_AWQ:
        return weights_scaling_factor is None
    return activation_scaling_factor is None or weights_scaling_factor is None


def quantize_linear(
    tensorrt_llm_layer, quantization: str, layer_config: Union[LinearConfig, QKVConfig]
):
    """Returns the quantized tensorrt_llm linear layer."""
    if quantization == QUANTIZATION_NONE:
        return tensorrt_llm_layer

    activation_scaling_factor = layer_config.activation_scaling_factor
    weights_scaling_factor = layer_config.weights_scaling_factor

    if quantization == QUANTIZATION_FP8:
        # FP8 is not sensitive to scaling factors. So we just quantize all layers possible.
        default_scaling_factor = torch.ones(1, dtype=torch.float32)
        if activation_scaling_factor is None:
            activation_scaling_factor = default_scaling_factor
        if weights_scaling_factor is None:
            weights_scaling_factor = default_scaling_factor

    if _should_skip_quantization(quantization, weights_scaling_factor, activation_scaling_factor):
        print(
            f"No valid scaling factors in {tensorrt_llm_layer._get_name()}, skipping quantization"
            " on this layer"
        )
        return tensorrt_llm_layer
    else:
        if quantization != QUANTIZATION_INT4_AWQ:
            assert torch.all(
                activation_scaling_factor > 0
            ), f"activation_scaling_factor {activation_scaling_factor} not positive"
        assert torch.all(
            weights_scaling_factor > 0
        ), f"weights_scaling_factor {weights_scaling_factor} not positive"

    bias = tensorrt_llm_layer.bias is not None

    extra_awq_args = {}
    if quantization == QUANTIZATION_INT4_AWQ:
        group_size = layer_config.awq_block_size
        extra_awq_args["group_size"] = group_size
        extra_awq_args["zero"] = False
        extra_awq_args["pre_quant_scale"] = True
    linear_layer_type = type(tensorrt_llm_layer)
    if linear_layer_type == Linear:
        if quantization in linear_qantized_trtllm_layer:
            linear = linear_qantized_trtllm_layer[quantization]
        else:
            assert False, f"{quantization} is not supported."
        quantized_linear_layer = linear(
            in_features=tensorrt_llm_layer.in_features,
            out_features=tensorrt_llm_layer.out_features * tensorrt_llm_layer.tp_size,
            bias=bias,
            dtype=tensorrt_llm_layer.dtype,
            tp_group=tensorrt_llm_layer.tp_group,
            tp_size=tensorrt_llm_layer.tp_size,
            gather_output=tensorrt_llm_layer.gather_output,
            **extra_awq_args,
        )
    elif linear_layer_type == RowLinear:
        if quantization in row_linear_qantized_trtllm_layer:
            row_linear = row_linear_qantized_trtllm_layer[quantization]
        else:
            assert False, f"{quantization} is not supported."
        quantized_linear_layer = row_linear(
            in_features=tensorrt_llm_layer.in_features * tensorrt_llm_layer.tp_size,
            out_features=tensorrt_llm_layer.out_features,
            bias=bias,
            dtype=tensorrt_llm_layer.dtype,
            tp_group=tensorrt_llm_layer.tp_group,
            tp_size=tensorrt_llm_layer.tp_size,
            **extra_awq_args,
        )
    else:
        assert False, f"{linear_layer_type} is not supported."

    # Adopt implementation from examples/llama/weight.py in the tekit repo for INT4 AWQ
    if quantization == QUANTIZATION_INT4_AWQ:
        preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
        pack_weight = preprocessor(
            layer_config.weight.view(torch.int8).T.contiguous().cpu(), torch.quint4x2
        ).view(torch.int8)

        pre_quant_scale = layer_config.prequant_scaling_factor
        scales = layer_config.weights_scaling_factor.T.contiguous()

        quantized_linear_layer.qweight.value = pack_weight
        quantized_linear_layer.scale.value = scales.to(torch.float16)
        quantized_linear_layer.pre_quant_scale.value = pre_quant_scale.reshape(1, -1).to(
            torch.float16
        )
        if bias:
            quantized_linear_layer.bias.value = layer_config.bias
    else:
        quantized_linear_layer.weight = tensorrt_llm_layer.weight
        quantized_linear_layer.bias = tensorrt_llm_layer.bias
        quantized_linear_layer.activation_scaling_factor.value = activation_scaling_factor
        quantized_linear_layer.weights_scaling_factor.value = weights_scaling_factor

        if hasattr(quantized_linear_layer, "prequant_scaling_factor"):
            prequant_scaling_factor = layer_config.prequant_scaling_factor
            quantized_linear_layer.prequant_scaling_factor.value = prequant_scaling_factor

    return quantized_linear_layer


def quantize_kv_cache(
    attention, quantization: str, kv_cache_scaling_factor: Optional[torch.Tensor]
):
    """Quantizes the kv cache in the attention layer.

    The FP8 KV cache is by-default enabled with scaling factor 1.
    The quantization is in place without a new tensorrt llm layer returned.
    """
    if quantization == QUANTIZATION_FP8:
        if kv_cache_scaling_factor is None:
            kv_cache_scaling_factor = torch.ones(1, dtype=torch.float32)

    if (
        kv_cache_scaling_factor is not None
        and attention.kv_orig_quant_scale is not None
        and attention.kv_quant_orig_scale is not None
    ):
        attention.kv_orig_quant_scale.value = 1 / kv_cache_scaling_factor
        attention.kv_quant_orig_scale.value = kv_cache_scaling_factor


def naive_quantization(config: ModelConfig, quantization: str):
    """Generates a constant scaling factor (1) with target quantization.

    This is for debugging and performance measurement only.
    """
    config.quantization = quantization
    # Here the scaling factor is not inversed.
    # In nvidia systems:
    # pytorch_quantization uses inv scale
    # onnx & trt uses non-inv scale
    # cask uses inv scale
    default_scaling_factor = torch.tensor(1, dtype=torch.float32)

    if quantization == QUANTIZATION_FP8:
        for layer in config.layers:
            linear_layers = [
                layer.attention.dense,
                layer.mlp.fc,
                layer.mlp.proj,
                layer.mlp.gate,
            ]

            if isinstance(layer.attention.qkv, QKVConfig):
                linear_layers += [
                    layer.attention.qkv.q,
                    layer.attention.qkv.k,
                    layer.attention.qkv.v,
                ]
            elif isinstance(layer.attention.qkv, LinearConfig):
                linear_layers += [layer.attention.qkv]

            for linear_layer in linear_layers:
                if linear_layer:
                    linear_layer.activation_scaling_factor = default_scaling_factor
                    linear_layer.weights_scaling_factor = default_scaling_factor

        if config.lm_head is not None:
            config.lm_head.activation_scaling_factor = default_scaling_factor
            config.lm_head.weights_scaling_factor = default_scaling_factor

    else:
        assert False, f"{quantization} not supported"
