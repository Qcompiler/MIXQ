# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The QWEN decoder implementation."""


from tensorrt_llm.layers import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.models.qwen.model import QWenBlock
from tensorrt_llm.quantization import QuantMode
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class QWENDecoderLayerBuilder(DecoderLayerBuilder):
    """The QWen implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        if self.quant_mode.has_fp8_kv_cache():
            # fp8_kv_cache does not work with Qwen now
            self.quant_mode = QuantMode(0)
        return QWenBlock(
            layer_id=self.layer_id,
            hidden_size=self.hidden_size,
            seq_length=layer.seq_length,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            apply_query_key_layer_scaling=False,
            hidden_act=self.hidden_act,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_scaling=None,
            quant_mode=self.quant_mode,
            # mlp_hidden_size = ffn_hidden_size * 2
            mlp_hidden_size=layer.ffn_hidden_size_local * 2 * self.tensor_parallel,
            neox_rotary_style=True,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            rms_norm_eps=layer.input_layernorm.eps,
        )
