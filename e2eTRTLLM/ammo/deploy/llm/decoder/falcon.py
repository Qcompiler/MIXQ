# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The Falcon decoder implementation.

Note: Current implementation only supports the 180B Falcon variant
"""


from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.models.falcon.model import FalconDecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class FalconDecoderLayerBuilder(DecoderLayerBuilder):
    """The Falcon implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        return FalconDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            hidden_act=non_gated_version(self.hidden_act),
            mlp_hidden_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            bias=layer.attention.dense.bias is not None,
            use_alibi=layer.use_alibi,
            new_decoder_architecture=layer.new_decoder_architecture,
            parallel_attention=layer.parallel_attention,
            layernorm_epsilon=layer.input_layernorm.eps,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            tp_rank=self.rank,
            layer_id=self.layer_id,
            quant_mode=self.quant_mode,
        )
