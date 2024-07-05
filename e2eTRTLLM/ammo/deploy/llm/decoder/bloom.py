# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The bloom decoder implementation."""

from argparse import Namespace

from tensorrt_llm.models.bloom.model import BloomDecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class BloomDecoderLayerBuilder(DecoderLayerBuilder):
    """The Bloom implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        config = Namespace()
        config.hidden_size = self.hidden_size
        config.dtype = self.dtype
        config.mapping = Namespace()
        config.mapping.tp_group = self.tp_group
        config.mapping.tp_size = self.tensor_parallel
        config.mapping.tp_rank = self.rank
        config.num_attention_heads = self.num_attention_heads
        config.num_key_value_heads = self.num_kv_heads
        config.num_hidden_layers = self.num_layers
        config.quant_mode = self.quant_mode
        config.intermediate_size = layer.ffn_hidden_size_local * self.tensor_parallel

        return BloomDecoderLayer(config, self.layer_id)
