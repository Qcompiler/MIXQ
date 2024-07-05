# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The BaiChuan decoder implementation."""


from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import (
    PositionEmbeddingType,
)
from tensorrt_llm.models.baichuan.model import BaichuanDecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class BaichuanDecoderLayerBuilder(DecoderLayerBuilder):
    """The BaiChuan implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        return BaichuanDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            position_embedding_type=(
                PositionEmbeddingType.alibi
                if layer.use_alibi
                else PositionEmbeddingType.rope_gpt_neox
            ),
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            hidden_act=non_gated_version(self.hidden_act),
            mlp_hidden_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            tp_rank=self.rank,
            quant_mode=self.quant_mode,
        )
