# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The LLAMA/LLAMA2 decoder implementation."""


from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import AttentionMaskType, MoeConfig, PositionEmbeddingType
from tensorrt_llm.models.llama.model import LLaMADecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class LLAMADecoderLayerBuilder(DecoderLayerBuilder):
    """The LLAMA implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        extra_args = {}
        if layer.rotary_base:
            extra_args = {"rotary_base": layer.rotary_base}

        return LLaMADecoderLayer(
            layer_id=self.layer_id,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            max_position_embeddings=self.max_position_embeddings,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            hidden_act=non_gated_version(self.hidden_act),
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            mlp_hidden_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            tp_rank=self.rank,
            quant_mode=self.quant_mode,
            rms_norm_eps=layer.input_layernorm.eps,
            attn_bias=False,
            mlp_bias=False,
            use_fused_mlp=False,
            moe_config=MoeConfig(),
            **extra_args,
        )
