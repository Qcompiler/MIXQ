# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for TensorRT-LLM checkpoint export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ammo import __version__

from .model_config import (
    LAYERNORM_RMS,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_NONE,
    QUANTIZATION_W4A8_AWQ,
    ModelConfig,
)

MODEL_NAME_TO_HF_ARCH_MAP = {
    "baichuan": "BaichuanForCausalLM",
    "chatglm": "ChatGLMForCausalLM",
    "falcon": "FalconForCausalLM",
    "gptj": "GPTJForCausalLM",
    "llama": "LlamaForCausalLM",
    "mpt": "MPTForCausalLM",
    "qwen": "QWenForCausalLM",
    "gemma": "GemmaForCausalLM",
    "bloom": "BloomForCausalLM",
}


def convert_to_tensorrt_llm_config(model_config: ModelConfig):
    """Covert to TensorRT-LLM checkpoint config."""
    decoder_type = model_config.layers[0].decoder_type
    tp_size = model_config.tensor_parallel
    pp_size = model_config.pipeline_parallel
    config = {
        "producer": {
            "name": "ammo",
            "version": __version__,
        },
        "architecture": MODEL_NAME_TO_HF_ARCH_MAP[decoder_type],
        "dtype": model_config.dtype,
        "num_hidden_layers": len(model_config.layers) * pp_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "hidden_size": model_config.hidden_size,
        "norm_epsilon": model_config.layers[0].input_layernorm.eps,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "hidden_act": model_config.hidden_act,
        "use_parallel_embedding": True,
        "embedding_sharding_dim": 0,
        "quantization": {"quant_algo": None, "kv_cache_quant_algo": None},
        "mapping": {
            "world_size": tp_size * pp_size,
            "tp_size": tp_size,
            "pp_size": pp_size,
        },
        "head_size": model_config.layers[0].attention_head_size,
        "intermediate_size": model_config.layers[0].ffn_hidden_size_local * tp_size,
    }

    if model_config.quantization == "fp8":
        config["quantization"].update({"quant_algo": "FP8"})
    elif model_config.quantization == "int4_awq":
        config["quantization"].update(
            {
                "quant_algo": "W4A16_AWQ",
                "group_size": model_config.layers[0].attention.qkv.awq_block_size,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["lm_head"],
            }
        )
    elif model_config.quantization == "w4a8_awq":
        config["quantization"].update(
            {
                "quant_algo": "W4A8_AWQ",
                "group_size": model_config.layers[0].attention.qkv.awq_block_size,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["lm_head"],
            }
        )
    elif model_config.quantization == "int8_sq":
        config["quantization"].update(
            {
                "quant_algo": "W8A8_SQ_PER_CHANNEL",
            }
        )
    elif model_config.quantization == QUANTIZATION_NONE:
        config["quantization"].update(
            {
                "quant_algo": None,
            }
        )
    else:
        config["quantization"].update(
            {
                "quant_algo": model_config.quantization,
            }
        )

    if decoder_type == "baichuan":
        config.update(
            {
                "position_embedding_type": (
                    "alibi" if model_config.layers[0].use_alibi else "rope_gpt_neox"
                ),
            }
        )
    elif decoder_type == "chatglm":
        config.update(
            {
                "position_embedding_type": "rope_gptj",
                "intermediate_size": model_config.layers[0].ffn_hidden_size_local * tp_size // 2,
                "max_position_embeddings": model_config.layers[0].seq_length,  # 32768
                "chatglm_version": model_config.layers[0].model_name.split("_")[0],
                "add_bias_linear": model_config.layers[0].attention.dense.bias is not None,  # False
                "add_qkv_bias": model_config.layers[0].attention.qkv.bias is not None,  # True
                "apply_query_key_layer_scaling": False,
                "apply_residual_connection_post_layernorm": model_config.layers[
                    0
                ].apply_residual_connection_post_layernorm,  # False
                "rmsnorm": (
                    model_config.layers[0].input_layernorm.layernorm_type == LAYERNORM_RMS
                ),  # True
                "rope_ratio": model_config.layers[0].rope_ratio,
            }
        )
    elif decoder_type == "falcon":
        config.update(
            {
                "position_embedding_type": (
                    "alibi_with_scale" if model_config.layers[0].use_alibi else "rope_gpt_neox"
                ),
                "bias": model_config.layers[0].attention.qkv.bias is not None,
                "parallel_attention": model_config.layers[0].parallel_attention,
                "new_decoder_architecture": model_config.layers[0].new_decoder_architecture,
            }
        )
    elif decoder_type == "gptj":
        config.update(
            {
                "position_embedding_type": "rope_gptj",
                "rotary_dim": model_config.layers[0].attention.rotary_dim,
            }
        )
    elif decoder_type == "llama":
        config.update(
            {
                "position_embedding_type": "rope_gpt_neox",
            }
        )
    elif decoder_type == "mpt":
        config.update(
            {
                "position_embedding_type": "alibi",
                "bias": model_config.layers[0].attention.qkv.bias is not None,
                "clip_qkv": model_config.layers[0].attention.clip_qkv,
            }
        )
    elif decoder_type == "qwen":
        config.update(
            {
                "position_embedding_type": "rope_gpt_neox",
                "rotary_emb_base": model_config.layers[
                    0
                ].rotary_base,  # Keep this for backward compatibility
                "rotary_pct": model_config.layers[0].rotary_pct,
                "intermediate_size": model_config.layers[0].ffn_hidden_size_local * 2 * tp_size,
            }
        )
    elif decoder_type == "gemma":
        config.update(
            {
                "position_embedding_type": "rope_gpt_neox",
            }
        )

    if model_config.layers[0].rotary_base:
        config.update(
            {
                "rotary_base": model_config.layers[0].rotary_base,
            }
        )

    if model_config.layers[0].attention.kv_cache_dtype is not None:
        config["quantization"].update(
            {
                "kv_cache_quant_algo": model_config.layers[0].attention.kv_cache_dtype,
            }
        )

    return config


def weights_to_npz(weights: Dict[str, np.ndarray], model_config: ModelConfig, export_dir: Path):
    """Export the model_config and the weights in the backward-compatible npz forward."""
    print("Warning: this is an old NPZ format and will be deprecated soon.")

    # step 1: rename key
    def get_npz_key(k):
        key_mapping = {
            "transformer.position_embedding": "_np:position_embedding:weight",
            "transformer.vocab_embedding": "_np:vocab_embedding:weight",
            "transformer.ln_f.weight": "_np:final_layernorm:weight",
            "transformer.ln_f.bias": "_np:final_layernorm:bias",
        }
        if k in key_mapping:
            return key_mapping[k]

        if "lm_head" in k:
            # src: lm_head.weight
            # dst: _np:lm_head:weight
            ns = k.split(".")
            return ":".join(["_np"] + ns)
        else:
            # src: transformers.layers.0.attention.q.weight
            # dst: _np:layers:20:attention:qkv:q:weight
            ns = k.split(".")
            return ":".join(["_np"] + ns[1:])

    # numpy doesn't know bfloat16, define abstract binary type instead

    np_bfloat16 = np.dtype("V2", metadata={"dtype": "bfloat16"})

    def _torch_to_numpy(x):
        if x.dtype != torch.bfloat16:
            return x.detach().cpu().numpy()
        return x.detach().view(torch.int16).cpu().numpy().view(np_bfloat16)

    np_weights = {}
    for k in list(weights):
        np_weights[get_npz_key(k)] = _torch_to_numpy(weights.pop(k))
    weights = np_weights

    # step 2: awq post process
    if (
        model_config.quantization == QUANTIZATION_INT4_AWQ
        or model_config.quantization == QUANTIZATION_W4A8_AWQ
    ):
        for k in list(weights):
            if k.endswith("weights_scaling_factor"):
                if "qkv" in k:
                    weights[k] = np.transpose(weights[k])
                else:
                    weights[k] = weights[k].flatten()

    weights_path = (
        export_dir
        / f"{model_config.layers[0].decoder_type}_tp{model_config.tensor_parallel}_rank{model_config.rank}.npz"
    )
    np.savez(weights_path, **weights)
