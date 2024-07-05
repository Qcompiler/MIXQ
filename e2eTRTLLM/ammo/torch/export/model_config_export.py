# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Code that export optimized models to dict representation of the ``ModelConfig``."""

import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Iterator, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file

from . import QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ
from .distribute import get_rank, get_world_size
from .layer_utils import (
    build_decoder_config,
    build_embedding_config,
    build_layernorm_config,
    build_linear_config,
    check_model_compatibility,
    get_transformer_layers,
    is_decoder_list,
    is_embedding,
    is_layernorm,
    is_linear,
)
from .model_config import CURRENT_VERSION, ModelConfig
from .model_config_utils import (
    merge_qkv,
    model_config_to_dict,
    pack_linear_weights,
    split_config_and_weights,
)
from .postprocess import (
    check_weight_shape_valid,
    pad_embedding_lm_head,
    postprocess_model_config,
    postprocess_tensors,
)
from .tensorrt_llm_utils import (
    convert_to_tensorrt_llm_config,
    weights_to_npz,
)


def torch_to_model_config(
    model: nn.Module,
    decoder_type: str,
    dtype: torch.dtype = torch.float16,
    inference_gpus: int = 0,
    inference_tensor_parallel: int = 0,
    inference_pipeline_parallel: int = 1,
    export_npz: bool = False,
) -> Iterator[ModelConfig]:
    """Converts the torch model to model_config per tensor parallel rank.

    ModelConfig is the representation that includes all the info needed for tensorrt_llm deployment.

    Args:
        model: the torch model.
        decoder_type: the type of the decoder, e.g. gpt2, gptj, llama or gptnext.
        dtype: the target weights data type for the export.
        inference_gpus (deprecate soon): same as inference_tensor_parallel.
        inference_tensor_parallel: The target inference time tensor parallel.
            We will merge or split the calibration tensor parallelism to inference.
            Default is 0, meaning using the calibration without manual config merge or split.
        inference_pipeline_parallel: The target inference time pipeline parallel.
        export_npz: Whether or not to export the model_config to the old NPZ format for backward
            compatibility.

    Yields:
        The ModelConfig of each GPU rank.
    """
    # Argument backward compatbility.
    if inference_tensor_parallel == 0 and inference_gpus > 0:
        print(
            "Warning: inference_gpus is going to be deprecated soon. Please use"
            " inference_tensor_parallel instead."
        )
        inference_tensor_parallel = inference_gpus

    if export_npz:
        print("Warning: export_npz is going to be deprecated soon and replaced by safetensors.")

    # Weights for mlp.fc in chatglm require specific processing for TP > 1.
    # Please refer the tekit example for more details and deployment instrcution
    if decoder_type == "chatglm" and inference_tensor_parallel > 1:
        raise NotImplementedError(
            "chatglm is not supported for TP > 1. Please refer to the example in TRTLLM."
        )

    if hasattr(model, "config"):
        # Huggingface or Megatron models
        model_metadata_config = model.config.__dict__

        if hasattr(model, "vocab_size"):
            # Megatron case
            vocab_size = model.vocab_size
        else:
            # Huggingface case
            vocab_size = model.config.vocab_size

        # For Baichuan 13B, we check if alibi is used with the alibi_mask property.
        if hasattr(model, "model") and hasattr(model.model, "alibi_mask"):
            model_metadata_config["alibi"] = True
    elif hasattr(model, "cfg"):
        # MegatronGPTModel
        model_metadata_config = dict(model.cfg)
        vocab_size = model.tokenizer.vocab_size
    else:
        raise ValueError("Cannot find valid model metadata config in model")

    training_pipeline_parallel = model_metadata_config.get("pipeline_model_parallel_size", 1)
    training_tensor_parallel = get_world_size() // training_pipeline_parallel
    model_metadata_config["training_pipeline_parallel"] = training_pipeline_parallel
    model_metadata_config["training_tensor_parallel"] = training_tensor_parallel

    if "make_vocab_size_divisible_by" in model_metadata_config:
        # For some nemo models, the vocab_size is pre-padded.
        # We calculate the pre-padded vocab_size with this config: make_vocab_size_divisible_by.
        make_vocab_size_divisible_by = model_metadata_config["make_vocab_size_divisible_by"]
        make_vocab_size_divisible_by_with_tp = (
            make_vocab_size_divisible_by * training_tensor_parallel
        )
        vocab_size = int(
            math.ceil(vocab_size / make_vocab_size_divisible_by_with_tp)
            * make_vocab_size_divisible_by_with_tp
        )
        print(
            f"the new vocab_size is updated: {vocab_size}, make_vocab_size_divisible_by"
            f" {make_vocab_size_divisible_by}, training_tensor_parallel"
            f" {training_tensor_parallel}."
        )

    transformer_layers = get_transformer_layers(model)
    if training_pipeline_parallel == 1:
        compatible, has_positional_embedding, has_embedding_layernorm = check_model_compatibility(
            transformer_layers
        )
    else:
        # For Megatron models with more than one PP,
        # we skip the compatibility check as not all ranks have the full model.
        compatible = len(transformer_layers) > 0
        has_positional_embedding = False
    assert compatible, "The model is not supported"

    config = ModelConfig(
        version=CURRENT_VERSION,
        dtype=str(dtype).split(".")[1],
        rank=get_rank(),
        tensor_parallel=get_world_size(),
        vocab_size=vocab_size,
    )
    # Build the full model_config dict layer by layer.
    for module in transformer_layers:
        if is_embedding(module):
            if config.vocab_embedding is None:
                # We assume the first embedding in the list the vocab_embedding.

                normalization_constant = 1
                # Normalize vocab embedding for gemma.
                if decoder_type == "gemma":
                    normalization_constant = model_metadata_config["hidden_size"] ** 0.5

                config.vocab_embedding = build_embedding_config(
                    module, dtype, normalization_constant=normalization_constant
                )
            elif has_positional_embedding:
                config.positional_embedding = build_embedding_config(module, dtype)
        elif is_decoder_list(module):
            layers = []
            for layer in module.children():
                layers.append(
                    build_decoder_config(layer, model_metadata_config, decoder_type, dtype)
                )
            config.layers = layers
        elif is_layernorm(module):
            if has_embedding_layernorm and config.ln_embed is None:
                # Assume embedding_layernorm is placed before the final_layernorm.
                config.ln_embed = build_layernorm_config(module, dtype)
            else:
                config.ln_f = build_layernorm_config(module, dtype)
        elif is_linear(module):
            config.lm_head = build_linear_config(module, "column", dtype)

    # For the training time PP, not all ranks will have the lm_head layer.
    if config.lm_head is None and training_pipeline_parallel == 1:
        # Models that share weights for lm_head and vocab_embedding
        assert decoder_type in [
            "mpt",
            "gpt2",
        ], f"lm_head not available for decoder {decoder_type}"
        config.share_embedding_table = True

    config.quantization = config.layers[0].quantization
    if config.quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
        if config.vocab_size % 64 != 0:
            assert training_tensor_parallel == 1, "We do not support padding for training time TP"
            print("Padding vocab_embedding and lm_head for AWQ weights export")
            pad_embedding_lm_head(config)

    check_weight_shape_valid(
        config,
        inference_tensor_parallel,
        training_tensor_parallel,
    )

    # If inference_tensor_parallel is different from world_size,
    # we try to merge or split the model configs based on the rank selected.
    # During exporting, we skip the ranks merged already
    # and only focus on the ranks matching inference_tensor_parallel.
    if inference_tensor_parallel > 0 or inference_pipeline_parallel > 0:
        model_configs = postprocess_model_config(
            config,
            inference_tensor_parallel,
            inference_pipeline_parallel,
            training_pipeline_parallel=training_pipeline_parallel,
        )
    else:
        model_configs = [config]

    for model_config in model_configs:
        assert model_config.rank >= 0, "Invalid model_config, postprocess_model_config fails."
        if export_npz:
            # The npz format is not compatible with ammo.deploy.llm for AWQ.
            model_config.version = 0.8
        else:
             


            merge_qkv(model_config)
            pack_linear_weights(model_config)
            # Postprocess the tensors in the model_config.
            # Exporting the safetensors also allows the tensor to be a view.
            postprocess_tensors(
                model_config, force_cpu=True, force_contiguous=True, force_non_view=False
            )
        yield model_config


def export_model_config(
    model: nn.Module,
    decoder_type: str,
    dtype: torch.dtype = torch.float16,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
    inference_gpus: int = 0,
    inference_tensor_parallel: int = 0,
    inference_pipeline_parallel: int = 1,
    export_tensorrt_llm_config: bool = False,
    export_npz: bool = False,
):
    """Exports the torch model to model_config and save to the export_dir.

    Args:
        model: the torch model.
        decoder_type: the type of the decoder, e.g. gpt2, gptj, llama or gptnext.
        dtype: the target weights data type for the export.
        export_dir: the target export path.
        inference_gpus (deprecate soon): same as inference_tensor_parallel.
        inference_tensor_parallel: The target inference time tensor parallel.
            We will merge or split the calibration tensor parallelism to inference.
            Default is 0, meaning using the calibration without manual config merge or split.
        inference_pipeline_parallel: The target inference time pipeline parallel.
        export_tensorrt_llm_config: Whether or not to export TensorRT-LLM checkpoint config file
            and postprocess weights to fully match the requirements of TensorRT-LLM checkpoint.
            If so, the exported checkpoint can be directly used to build TensorRT-LLM engines.
            See https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/new_workflow.md for
            more details of TensorRT-LLM checkpoint format.
        export_npz: Whether or not to export the model_config to the old NPZ format for backward
            compatibility.

    For tensorrt_llm deployment, save the representation under ``export_dir``.
    We will save the model_config as two files:

        * ``.json``: The nested dict without the weights.
        * ``.safetensors``: The file for the list of weights as safetensors. Unique for each rank.
    """
    # Argument backward compatbility.
    if inference_tensor_parallel == 0 and inference_gpus > 0:
        print(
            "Warning: inference_gpus is going to be deprecated soon. Please use"
            " inference_tensor_parallel instead."
        )
        inference_tensor_parallel = inference_gpus

    if inference_pipeline_parallel > 1:
        assert export_tensorrt_llm_config, (
            "Inference time pipeline parallel is only supported with export_tensorrt_llm_config on"
            " and the build API from the TensorRT LLM repo"
        )

    assert not (
        export_npz and export_tensorrt_llm_config
    ), "Cannot set both tensorrt_llm_config and export_npz at the same time."

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    try:
        for model_config in torch_to_model_config(
            model=model,
            decoder_type=decoder_type,
            dtype=dtype,
            inference_tensor_parallel=inference_tensor_parallel,
            inference_pipeline_parallel=inference_pipeline_parallel,
            export_npz=export_npz,
        ):
            weights = {}
            model_config_dict = model_config_to_dict(model_config)
            # We split the weights from model_config and save them separately as two files.
            split_config_and_weights(model_config_dict, weights)

            
            rank = model_config.rank
            if rank == 0:
                # We only export the json once across ranks as all jsons should be the same except for the rank.
                if export_tensorrt_llm_config:
                    tensorrt_llm_config = convert_to_tensorrt_llm_config(model_config)
                    with open(export_dir / "config.json", "w") as f:
                        json.dump(tensorrt_llm_config, f, indent=4)
                else:
                    json_path = export_dir / f"{decoder_type}_tp{inference_tensor_parallel}.json"
                    with open(json_path, "w") as f:
                        json.dump(model_config_dict, f, indent=4)

            if export_tensorrt_llm_config:
                weights_path = export_dir / f"rank{rank}.safetensors"
                save_file(weights, weights_path)
            elif export_npz:
                weights_to_npz(weights, model_config, export_dir)
            else:
                weights_path = export_dir / f"rank{rank}.safetensors"
                save_file(weights, weights_path)

    except Exception as e:
        fallback_model_path = export_dir / f"ammo_model.{get_rank()}.pth"
        torch.save(model.state_dict(), fallback_model_path)
        print(
            "Cannot export model to the model_config. The AMMO optimized model state_dict"
            f" (including the quantization factors) is saved to {fallback_model_path} using"
            " torch.save for further inspection."
        )
        print(f"Detailed export error: {e}")
        traceback.print_exc()
