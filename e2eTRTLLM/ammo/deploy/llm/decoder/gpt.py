# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The GPT2 decoder implementation."""


import inspect

from tensorrt_llm.layers import AttentionMaskType, MoeConfig, PositionEmbeddingType
from tensorrt_llm.models.gpt.model import GPTDecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class GPTDecoderLayerBuilder(DecoderLayerBuilder):
    """The GPT implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_pct = layer.rotary_pct
        position_embedding_type = (
            PositionEmbeddingType.learned_absolute
            if rotary_pct == 0.0
            else PositionEmbeddingType.rope_gpt_neox
        )

        bias_qkv = layer.attention.qkv.bias is not None

        # Supporting different attention_head_size per layer. This feature has yet
        # to be exposed through TensorRT-LLM's GPTDecoderLayer yet. The WAR is to
        # inspect the signature.
        additional_kwargs = {}
        if "attention_head_size" in inspect.signature(GPTDecoderLayer).parameters:
            additional_kwargs = {"attention_head_size": self.attention_head_size}

        return GPTDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            dtype=self.dtype,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            hidden_act=self.hidden_act,
            position_embedding_type=position_embedding_type,
            rotary_embedding_percentage=rotary_pct,
            inter_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            bias=bias_qkv,
            num_kv_heads=self.num_kv_heads,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            tp_rank=self.rank,
            quant_mode=self.quant_mode,
            instance_id=self.layer_id,
            moe_config=MoeConfig(),
            **additional_kwargs,
        )
