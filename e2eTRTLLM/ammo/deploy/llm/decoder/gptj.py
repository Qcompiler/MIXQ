# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The GPTJ decoder implementation."""


from tensorrt_llm.models.gptj.model import GPTJDecoderLayer
from typing_extensions import override

from .decoder import DecoderLayerBuilder


class GPTJDecoderLayerBuilder(DecoderLayerBuilder):
    """The GPTJ implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        return GPTJDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rotary_dim=layer.attention.rotary_dim,
            dtype=self.dtype,
            hidden_act=self.hidden_act,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            quant_mode=self.quant_mode,
        )
