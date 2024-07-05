# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The utils to support nemo models."""

from pathlib import Path
from typing import Dict

from transformers import GPT2Tokenizer, PreTrainedTokenizer, T5Tokenizer


def _build_tokenizer(tokenizer_config: Dict):
    if tokenizer_config["library"] == "sentencepiece":
        # AMMO modification.
        # Turn off legacy model by default: See https://github.com/huggingface/transformers/pull/24622
        tokenizer = T5Tokenizer(tokenizer_config["model"], extra_ids=0, legacy=False)
    elif "GPT2" in tokenizer_config["type"]:
        tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
    else:
        raise ValueError(f'Tokenizer type {tokenizer_config["library"]} not handled')

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    return tokenizer


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    model_path = (
        tokenizer_dir_or_path / "tokenizer.model"
        if tokenizer_dir_or_path.is_dir()
        else tokenizer_dir_or_path
    )
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return _build_tokenizer(tokenizer_config)
