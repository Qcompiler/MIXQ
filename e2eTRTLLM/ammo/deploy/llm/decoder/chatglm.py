# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The ChatGlm decoder implementation."""

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.chatglm.model import ChatGLMDecoderLayer, ChatGLMParams
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class ChatGlmDecoderLayerBuilder(DecoderLayerBuilder):
    """The ChatGlm implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        assert layer.input_layernorm.eps == layer.post_layernorm.eps
        assert layer.input_layernorm.layernorm_type == layer.post_layernorm.layernorm_type
        config = ChatGLMParams(
            apply_query_key_layer_scaling=False,
            apply_residual_connection_post_layernorm=layer.apply_residual_connection_post_layernorm,
            dtype=self.dtype,
            enable_debug_output=False,
            ffn_hidden_size=layer.ffn_hidden_size_local // 2 * self.tensor_parallel,
            hidden_act=layer.mlp.hidden_act,
            hidden_size=layer.hidden_size,
            linear_bias=layer.attention.dense.bias is not None,
            mapping=Mapping(
                world_size=self.tensor_parallel,
                rank=self.rank,
                tp_size=self.tensor_parallel,
            ),
            model_name=layer.model_name,
            norm_epsilon=layer.input_layernorm.eps,
            num_heads=layer.num_attention_heads,
            num_kv_heads=layer.num_kv_heads,
            num_layers=self.num_layers,
            qkv_bias=layer.attention.qkv.bias is not None,
            quant_mode=self.quant_mode,
            rmsnorm=layer.input_layernorm.layernorm_type == "rms",
            use_cache=layer.use_cache,
        )

        return ChatGLMDecoderLayer(self.layer_id, config)
