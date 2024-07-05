# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This module defines the model_config format.

This format can be converted from huggingface, nemo or AMMO quantized model.
And we will build tensorrt_llm engine from the context saved with this format.
"""

import math
from dataclasses import dataclass, field
from typing import List, Union

import torch

DECODER_GPT2 = "gpt2"
DECODER_GPTJ = "gptj"
DECODER_LLAMA = "llama"
DECODER_GPTNEXT = "gptnext"
DECODER_FALCON = "falcon"
DECODER_BAICHUAN = "baichuan"
DECODER_MPT = "mpt"
DECODER_BLOOM = "bloom"
DECODER_CHATGLM = "chatglm"
DECODER_QWEN = "qwen"

QUANTIZATION_NONE = ""
QUANTIZATION_FP8 = "fp8"
QUANTIZATION_INT8_SQ = "int8_sq"
QUANTIZATION_INT4_AWQ = "int4_awq"
QUANTIZATION_W4A8_AWQ = "w4a8_awq"
QUANTIZATION_INT8_Mix = "int8_mix"
KV_CACHE_FP8 = "FP8"
KV_CACHE_INT8 = "INT8"

LINEAR_COLUMN = "column"
LINEAR_ROW = "row"

LAYERNORM_DEFAULT = ""
LAYERNORM_RMS = "rms"

SUPPORTED_VERSION = 0.2
AWQ_SUPPORTED_VERSION = 0.9
CURRENT_VERSION = 1.4


@dataclass
class EmbeddingConfig:
    """The embedding layer config."""

    weight: torch.Tensor = None

    @property
    def local_vocab_size(self):
        """Infers the vocab_size from the embedding layer weights shape."""
        return self.weight.shape[0]

    @property
    def hidden_size(self):
        """Infers the hidden_size from the embedding layer weights shape."""
        return self.weight.shape[1]


@dataclass
class LayernormConfig:
    """The layernorm layer config."""

    weight: torch.Tensor = None
    bias: torch.Tensor = None
    layernorm_type: str = LAYERNORM_DEFAULT
    eps: float = 1e-5


@dataclass
class LinearConfig:
    """The linear layer config."""

    linear_type: str = LINEAR_COLUMN
    weight: torch.Tensor = None
    bias: torch.Tensor = None
    activation_scaling_factor: torch.Tensor = None
    weights_scaling_factor: torch.Tensor = None

    # For methods like W4A8 AWQ, we have two quantizers for weights
    # For W4A8, the first quantizer is for INT4 quantization and the second quantizer is for FP8 quantization
    # `weight_scaling_factor_2` is the scaling factor the the second FP8 quantizer
    weights_scaling_factor_2: torch.Tensor = None

    prequant_scaling_factor: torch.Tensor = None

    # for Mix precision
    fp_ind : torch.Tensor = None
    fp_weight : torch.Tensor = None
    qweight : torch.Tensor = None
    qzeros : torch.Tensor = None
    scales : torch.Tensor = None


    awq_block_size: int = 0


@dataclass
class QKVConfig:
    """The QKV layer config."""

    q: LinearConfig = None
    k: LinearConfig = None
    v: LinearConfig = None

    @property
    def weight(self):
        """The generated linear layer weight.

        The Q, K, V weights are concat together to fit the TensorRT-LLM QKV linear layer.
        """
        return torch.cat((self.q.weight, self.k.weight, self.v.weight))

    @property
    def bias(self):
        """The generated linear layer bias.

        The Q, K, V bias are concat together to fit the TensorRT-LLM QKV linear layer.
        """
        if self.q.bias is None:
            assert (
                self.k.bias is None and self.v.bias is None
            ), "K and V should have valid bias as Q"
            return None
        return torch.cat((self.q.bias, self.k.bias, self.v.bias))

    @property
    def activation_scaling_factor(self):
        """Returns the merged activation_scaling_factor across Q, K and V.

        The max of the Q, K, V activation scaling factors is returned.
        """
        if (
            self.q.activation_scaling_factor is None
            or self.k.activation_scaling_factor is None
            or self.v.activation_scaling_factor is None
        ):
            return None

        return (
            torch.stack(
                [
                    self.q.activation_scaling_factor,
                    self.k.activation_scaling_factor,
                    self.v.activation_scaling_factor,
                ]
            )
            .max(dim=0)
            .values
        )

    @property
    def weights_scaling_factor(self):
        """Returns the merged weights_scaling_factor across Q, K and V.

        If the quantization is FP8, the max of the Q, K, V weight scaling factors is returned.
        If the quanitzation is INT8_SQ, the concat value is returned.
        """
        if (
            self.q.weights_scaling_factor is None
            or self.k.weights_scaling_factor is None
            or self.v.weights_scaling_factor is None
        ):
            return None

        if self.q.weights_scaling_factor.numel() != 1:
            # for Int4 AWQ and Int8 SQ case, we concatenate the
            # q_weight_scaling_factor, k_weight_scaling_factor, v_weight_scaling_factor
            qkv_weights_scaling_factor = torch.cat(
                (
                    self.q.weights_scaling_factor,
                    self.k.weights_scaling_factor,
                    self.v.weights_scaling_factor,
                )
            )
        else:
            # for FP8 set qkv_weight_scaling_factor to the max of
            # q_weight_scaling_factor, k_weight_scaling_factor, v_weight_scaling_factor
            qkv_weights_scaling_factor = (
                torch.stack(
                    [
                        self.q.weights_scaling_factor,
                        self.k.weights_scaling_factor,
                        self.v.weights_scaling_factor,
                    ],
                )
                .max(dim=0)
                .values
            )
        return qkv_weights_scaling_factor

    @property
    def weights_scaling_factor_2(self):
        """Returns the merged weights_scaling_factor_2 across Q, K and V.

        weight_scaling_factor_2 is needed for W4A8 AWQ.
        """
        if (
            self.q.weights_scaling_factor_2 is None
            or self.k.weights_scaling_factor_2 is None
            or self.v.weights_scaling_factor_2 is None
        ):
            return None

        # For W4A8 AWQ, weight_scaling_factor_2 corresponds to the per-tensor FP8 quantization.
        # Hence weight_scaling_factor_2 should be a scalar.
        assert self.q.weights_scaling_factor_2.numel() == 1

        # set qkv_weight_scaling_factor_2 to the max of q,k,v weight_scaling_factor_2
        qkv_weights_scaling_factor_2 = (
            torch.stack(
                [
                    self.q.weights_scaling_factor_2,
                    self.k.weights_scaling_factor_2,
                    self.v.weights_scaling_factor_2,
                ]
            )
            .max(dim=0)
            .values
        )

        return qkv_weights_scaling_factor_2

    @property
    def prequant_scaling_factor(self):
        """Returns the merged prequant_scaling_factor across Q, K and V.

        Prequant scaling factors for Q, K, V should be the same. So just return one of them.
        """
        if (
            self.q.prequant_scaling_factor is None
            or self.k.prequant_scaling_factor is None
            or self.v.prequant_scaling_factor is None
        ):
            return None

        assert torch.equal(
            self.q.prequant_scaling_factor, self.k.prequant_scaling_factor
        ) and torch.equal(
            self.k.prequant_scaling_factor, self.v.prequant_scaling_factor
        ), "Prequant scaling factors of Q, K and V should be the same"
        return self.q.prequant_scaling_factor

    @property
    def awq_block_size(self):
        """Returns the awq_block_size of this QKV layer."""
        assert (
            self.q.awq_block_size == self.k.awq_block_size == self.v.awq_block_size
        ), "awq_block_size of QKV should be the same."
        return self.q.awq_block_size


@dataclass
class AttentionConfig:
    """The attention layer config."""

    # QKV can either be stored as splitted (for easier postprocessing)
    # or merged (for TRT LLM export)
    qkv: Union[QKVConfig, LinearConfig] = None
    dense: LinearConfig = None
    kv_cache_scaling_factor: torch.Tensor = None
    kv_cache_dtype: str = None

    rotary_dim: int = -math.inf
    # MPT variants
    clip_qkv: float = None


@dataclass
class MLPConfig:
    """The MLP layer config."""

    fc: LinearConfig = None
    gate: LinearConfig = None
    proj: LinearConfig = None
    hidden_act: str = ""


@dataclass
class DecoderLayerConfig:
    """The decoder layer config."""

    quantization: str = QUANTIZATION_NONE

    decoder_type: str = ""
    input_layernorm: LayernormConfig = None
    mlp_layernorm: LayernormConfig = None
    attention: AttentionConfig = None
    post_layernorm: LayernormConfig = None
    mlp: MLPConfig = None

    num_attention_heads: int = 0
    # Supporting different attention_head_size per layer.
    attention_head_size: int = None

    num_kv_heads: int = 0
    max_position_embeddings: int = 0
    rotary_pct: float = 0

    # Falcon and Baichuan variants
    use_alibi: bool = False
    new_decoder_architecture: bool = False
    parallel_attention: bool = False

    # chatglm variants
    apply_residual_connection_post_layernorm: bool = False
    use_cache: bool = True
    model_name: str = ""
    rope_ratio: float = 1.0

    # Qwen config
    seq_length: int = 0

    # Qwen and CodeLlama
    rotary_base: int = 0

    @property
    def hidden_size(self):
        """Returns the hidden size of the transformer model."""
        return self.mlp.fc.weight.shape[1]

    @property
    def ffn_hidden_size_local(self):
        """Returns the ffn hidden size of the transformer model."""
        if self.quantization not in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
            return self.mlp.fc.weight.shape[0]
        return self.mlp.fc.weight.shape[0] * 2


@dataclass
class ModelConfig:
    """The full LLM model config that includes the full information needed for tensorrt_llm engine building.

    This class includes all the fields that tensorrt_llm supports, but not all of the fields are required.
    pipeline_parallel > 1 is only supported for TensorRT-LLM checkpoint.
    """

    version: float = 0.0

    # Global metadata
    quantization: str = QUANTIZATION_NONE
    dtype: str = "float16"
    vocab_size: int = 0

    # Parallel metadata
    rank: int = 0
    tensor_parallel: int = 1
    pipeline_parallel: int = 1

    # Model structure and weights
    vocab_embedding: EmbeddingConfig = None
    positional_embedding: EmbeddingConfig = None
    ln_embed: LayernormConfig = None
    layers: List[DecoderLayerConfig] = field(default_factory=list)

    # Deprecated, will be replaced by ln_f to match tensorrt_llm
    final_layernorm: LayernormConfig = None

    ln_f: LayernormConfig = None

    lm_head: LinearConfig = None
    share_embedding_table: bool = False

    @property
    def vocab_size_padded(self):
        """Returns the padded vocab_size of the model rounds to the tensor_parallel."""

        def _pad_vocab_size(vocab_size, tp_size):
            return int(math.ceil(vocab_size / tp_size) * tp_size)

        return _pad_vocab_size(self.vocab_size, self.tensor_parallel)

    @property
    def hidden_size(self):
        """Returns the hidden_size of the model."""
        return self.vocab_embedding.hidden_size

    @property
    def max_position_embeddings(self):
        """Returns the max_position_embedding of the model."""
        return self.layers[0].max_position_embeddings

    @property
    def num_attention_heads(self):
        """Returns the num_attention_heads of the model."""
        return self.layers[0].num_attention_heads

    @property
    def num_kv_heads(self):
        """Returns the num_key_value_heads of the model."""
        return (
            self.layers[0].num_kv_heads
            if self.layers[0].num_kv_heads > 0
            else self.num_attention_heads
        )

    @property
    def hidden_act(self):
        """Returns the hidden_act of the model."""
        return self.layers[0].mlp.hidden_act
