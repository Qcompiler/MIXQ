# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The MPT decoder implementation."""


from tensorrt_llm.layers import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.models.gpt.model import GPTDecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class MPTDecoderLayerBuilder(DecoderLayerBuilder):
    """The MPT implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_pct = layer.rotary_pct
        has_qkv_bias = layer.attention.qkv.bias is not None

        return GPTDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            dtype=self.dtype,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            hidden_act=self.hidden_act,
            position_embedding_type=PositionEmbeddingType["alibi"],
            rotary_embedding_percentage=rotary_pct,
            inter_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            bias=has_qkv_bias,
            num_kv_heads=self.num_kv_heads,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            tp_rank=self.rank,
            quant_mode=self.quant_mode,
            instance_id=self.layer_id,
        )
