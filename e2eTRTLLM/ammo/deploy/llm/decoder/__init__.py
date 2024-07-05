# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LLM Decoder implementation for tensorrt_llm conversion."""
from typing import Dict, Type

import tensorrt as trt

from ammo.torch.export import (
    DECODER_BAICHUAN,
    DECODER_BLOOM,
    DECODER_CHATGLM,
    DECODER_FALCON,
    DECODER_GPT2,
    DECODER_GPTJ,
    DECODER_GPTNEXT,
    DECODER_LLAMA,
    DECODER_MPT,
    DECODER_QWEN,
    QUANTIZATION_NONE,
)

from .baichuan import BaichuanDecoderLayerBuilder
from .bloom import BloomDecoderLayerBuilder
from .chatglm import ChatGlmDecoderLayerBuilder
from .decoder import DecoderLayerBuilder
from .falcon import FalconDecoderLayerBuilder
from .gpt import GPTDecoderLayerBuilder
from .gptj import GPTJDecoderLayerBuilder
from .llama import LLAMADecoderLayerBuilder
from .mpt import MPTDecoderLayerBuilder
from .qwen import QWENDecoderLayerBuilder

DECODER_REGISTRY: Dict[str, Type[DecoderLayerBuilder]] = {
    DECODER_GPT2: GPTDecoderLayerBuilder,
    DECODER_GPTJ: GPTJDecoderLayerBuilder,
    DECODER_LLAMA: LLAMADecoderLayerBuilder,
    DECODER_GPTNEXT: GPTDecoderLayerBuilder,
    DECODER_FALCON: FalconDecoderLayerBuilder,
    DECODER_BAICHUAN: BaichuanDecoderLayerBuilder,
    DECODER_MPT: MPTDecoderLayerBuilder,
    DECODER_BLOOM: BloomDecoderLayerBuilder,
    DECODER_CHATGLM: ChatGlmDecoderLayerBuilder,
    DECODER_QWEN: QWENDecoderLayerBuilder,
}


def build_decoder_layer(
    layer,
    layer_id: int,
    num_layers: int,
    dtype=trt.float16,
    quantization=QUANTIZATION_NONE,
    rank=0,
    tensor_parallel=1,
):
    """Builds the tensorrt llm decoder layer module with the layer config as the input."""
    assert layer.decoder_type in DECODER_REGISTRY, f"{layer.decoder_type} not supported"
    builder = DECODER_REGISTRY[layer.decoder_type]
    decoder_builder = builder(
        layer, layer_id, num_layers, dtype, quantization, rank, tensor_parallel
    )
    return decoder_builder.decoder
