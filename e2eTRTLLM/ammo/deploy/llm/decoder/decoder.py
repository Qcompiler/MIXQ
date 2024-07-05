# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The parent decoder class implementation for model_config and tensorrt_llm conversion."""

from abc import ABC, abstractmethod

import tensorrt as trt
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.quantization import QuantMode

from ammo.torch.export import (
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_NONE,
    DecoderLayerConfig,
    from_quantized_weight,
)

from ..quantization_utils import quantize_kv_cache, quantize_linear
from ..tensorrt_llm_utils import (
    get_tensor_parallel_group,
)


class DecoderLayerBuilder(ABC):
    """An abstracted transformer decoder layer with tensorrt_llm implementation taking DecoderLayerConfig as the input.

    Individual decoder layers are supposed to extend this class and implement the customized
    abstracted method.
    """

    @abstractmethod
    def build_decoder(self, layer):
        """Returns the built decoder layer."""
        pass

    def __init__(
        self,
        layer: DecoderLayerConfig,
        layer_id: int,
        num_layers: int,
        dtype: trt.DataType = trt.float16,
        quantization: str = QUANTIZATION_NONE,
        rank: int = 0,
        tensor_parallel: int = 1,
    ):
        """Initializes the DecoderLayer."""
        super().__init__()
        assert isinstance(dtype, trt.DataType)
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dtype = dtype
        self.quantization = quantization
        self.rank = rank
        self.tensor_parallel = tensor_parallel
        self.tp_group = get_tensor_parallel_group(tensor_parallel)

        self.hidden_size = layer.hidden_size
        self.num_attention_heads = layer.num_attention_heads
        self.num_kv_heads = (
            layer.num_kv_heads if layer.num_kv_heads > 0 else layer.num_attention_heads
        )
        # Supporting different attention_head_size per layer.
        self.attention_head_size = layer.attention_head_size

        assert (
            self.num_attention_heads % self.num_kv_heads
        ) == 0, "MQA/GQA requires the number of heads to be divisible by the number of K/V heads."
        assert (self.num_kv_heads % self.tensor_parallel) == 0 or (
            self.tensor_parallel % self.num_kv_heads
        ) == 0, (
            "MQA/GQA requires either the number of K/V heads to be divisible by the number of GPUs"
            " OR the number of GPUs to be divisible by the number of K/V heads."
        )

        self.max_position_embeddings = layer.max_position_embeddings
        self.hidden_act = layer.mlp.hidden_act
        self.quant_mode = QuantMode(0)
        if self.quantization == QUANTIZATION_FP8:
            self.quant_mode = self.quant_mode.set_fp8_kv_cache()
        elif (
            self.quantization == QUANTIZATION_INT8_SQ
            and layer.attention.kv_cache_scaling_factor is not None
        ):
            self.quant_mode = self.quant_mode.set_int8_kv_cache()
        elif self.quantization == QUANTIZATION_INT4_AWQ:
            self.quant_mode = QuantMode.from_description(
                quantize_weights=True, per_group=True, use_int4_weights=True
            )
            block_size = layer.attention.qkv.awq_block_size
            if self.hidden_size % block_size != 0:
                raise ValueError(
                    f"Model not supported with AWQ. Hidden size {self.hidden_size} is not a"
                    f" multiple of group size {block_size}."
                )

        self.decoder = self.build_decoder(layer)
        self.assign_weights(layer)
        self.quantize(layer)

    def assign_weights(self, layer: DecoderLayerConfig):
        """Assign the weights to the attention tensorrt_llm layer."""
        # Locate input layernorm
        input_layernorm = None
        for k in ["input_layernorm", "ln_1", "pre_norm"]:
            if hasattr(self.decoder, k):
                input_layernorm = getattr(self.decoder, k)

        if input_layernorm:
            input_layernorm.weight.value = layer.input_layernorm.weight
            if layer.input_layernorm.bias is not None:
                input_layernorm.bias.value = layer.input_layernorm.bias
            # Bias of input layernorm could be disabled for some models, e.g., MPT, filling zeros in this case.
            elif hasattr(input_layernorm, "bias") and input_layernorm.bias is not None:
                assert layer.decoder_type in [
                    "mpt"
                ], f"Input layernorm bias not available in {layer.decoder_type}."
                input_layernorm.bias._value.fill(0)

        if layer.mlp_layernorm is not None:
            self.decoder.mlp_layernorm.weight.value = layer.mlp_layernorm.weight
            if layer.mlp_layernorm.bias is not None:
                self.decoder.mlp_layernorm.bias.value = layer.mlp_layernorm.bias

        # For int4_awq, weights will be set at the quantize function.
        if self.quantization != QUANTIZATION_INT4_AWQ:
            torch_dtype = trt_dtype_to_torch(self.dtype)
            self.decoder.attention.qkv.weight.value = from_quantized_weight(
                layer.attention.qkv.weight,
                layer.attention.qkv.weights_scaling_factor,
                self.quantization,
                torch_dtype,
            )
            self.decoder.attention.dense.weight.value = from_quantized_weight(
                layer.attention.dense.weight,
                layer.attention.dense.weights_scaling_factor,
                self.quantization,
                torch_dtype,
            )
            self.decoder.mlp.fc.weight.value = from_quantized_weight(
                layer.mlp.fc.weight,
                layer.mlp.fc.weights_scaling_factor,
                self.quantization,
                torch_dtype,
            )
            self.decoder.mlp.proj.weight.value = from_quantized_weight(
                layer.mlp.proj.weight,
                layer.mlp.proj.weights_scaling_factor,
                self.quantization,
                torch_dtype,
            )
            if layer.mlp.gate:
                self.decoder.mlp.gate.weight.value = from_quantized_weight(
                    layer.mlp.gate.weight,
                    layer.mlp.gate.weights_scaling_factor,
                    self.quantization,
                    torch_dtype,
                )

        if layer.attention.qkv.bias is not None:
            self.decoder.attention.qkv.bias.value = layer.attention.qkv.bias
        if self.decoder.attention.dense.bias is not None:
            self.decoder.attention.dense.bias.value = layer.attention.dense.bias

        if layer.post_layernorm is not None:
            # Locate post layernorm
            post_layernorm = None
            for k in ["post_layernorm", "ln_2", "post_norm"]:
                if hasattr(self.decoder, k):
                    post_layernorm = getattr(self.decoder, k)

            if post_layernorm:
                post_layernorm.weight.value = layer.post_layernorm.weight
                if layer.post_layernorm.bias is not None:
                    post_layernorm.bias.value = layer.post_layernorm.bias
                # Bias of post layernorm could be disabled for some models, e.g., MPT, filling zeros in this case.
                elif hasattr(post_layernorm, "bias") and post_layernorm.bias is not None:
                    assert layer.decoder_type in [
                        "mpt"
                    ], f"Input layernorm bias not available in {layer.decoder_type}."
                    post_layernorm.bias._value.fill(0)

        bias = layer.mlp.fc.bias is not None
        if bias:
            self.decoder.mlp.fc.bias.value = layer.mlp.fc.bias
            self.decoder.mlp.proj.bias.value = layer.mlp.proj.bias

        if layer.mlp.gate and bias:
            self.decoder.mlp.gate.bias.value = layer.mlp.gate.bias

    def quantize(self, layer: DecoderLayerConfig):
        """Quantizes the decoder layer based on the layer config."""
        quantize_kv_cache(
            self.decoder.attention, self.quantization, layer.attention.kv_cache_scaling_factor
        )
        self.decoder.attention.qkv = quantize_linear(
            self.decoder.attention.qkv, self.quantization, layer.attention.qkv
        )
        self.decoder.attention.dense = quantize_linear(
            self.decoder.attention.dense, self.quantization, layer.attention.dense
        )
        self.decoder.mlp.fc = quantize_linear(self.decoder.mlp.fc, self.quantization, layer.mlp.fc)
        self.decoder.mlp.proj = quantize_linear(
            self.decoder.mlp.proj, self.quantization, layer.mlp.proj
        )

        if hasattr(self.decoder.mlp, "gate"):
            self.decoder.mlp.gate = quantize_linear(
                self.decoder.mlp.gate, self.quantization, layer.mlp.gate
            )

        self.decoder.quant_mode = self.quant_mode
