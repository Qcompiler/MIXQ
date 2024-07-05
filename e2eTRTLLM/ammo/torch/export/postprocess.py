# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils to load and process model_config."""

import copy
from dataclasses import fields, is_dataclass
from typing import List

import torch

from .distribute import (
    get_configs_parallel,
    get_group,
    get_rank,
    get_tensors_parallel,
    get_world_size,
)
from .layer_utils import get_weights_scaling_factor, re_smooth_and_get_scale
from .model_config import LINEAR_COLUMN, EmbeddingConfig, LinearConfig, ModelConfig
from .model_config_utils import pad_weights


def _same_tensor(tensors: List[torch.Tensor]):
    return all(t is None for t in tensors) or all(torch.equal(tensors[0], t) for t in tensors[1:])


def _split_model_config_for_tp(merged_config, split_factor):
    """This method splits the tensor fields for linear config so the config can be used with more GPUs.

    The implementation is recursive.
    """
    configs = [copy.copy(merged_config) for _ in range(split_factor)]

    if isinstance(merged_config, EmbeddingConfig):
        weights = torch.chunk(pad_weights(merged_config.weight, split_factor), split_factor, dim=0)
        for i, config in enumerate(configs):
            config.weight = weights[i]

    elif isinstance(merged_config, LinearConfig):
        # The scaling factors split rule is summarized as below:

        # S: all ranks should have the same scaling factor.
        # P0: Split the scaling factors on dim 0.
        # P1: Split the scaling factors on dim 1.
        # NA: Not valid / present

        # ws: weight scaling factor
        # as: activation scaling factor
        # ps: prequant scaling factor

        # C: Colum Linear
        # R: Row Linear

        # F: FP8
        # I8: INT8 SQ
        # I4: INT4 AWQ

        # Split Rules:
        #      ws  as  ps
        # FC   S   S   NA
        # FR   S   S   NA
        # I8C  P0  S   S
        # I8R  S   S   P0
        # I4C  P0  S   S
        # I4R  P1  S   P0

        # For INT4 AWQ reference implemention: please check examples/llama/weight.py in the tekit repo
        # For normal linear layers, we split the column linear on the dim 0 and row on the dim 1
        split_axis = 0 if merged_config.linear_type == LINEAR_COLUMN else 1
        if merged_config.linear_type == LINEAR_COLUMN:
            merged_config.weight = pad_weights(merged_config.weight, split_factor)
        weights = torch.chunk(merged_config.weight, split_factor, dim=split_axis)
        for i, config in enumerate(configs):
            config.weight = weights[i]

        # Only split the bias for column linear.
        if merged_config.linear_type == LINEAR_COLUMN and merged_config.bias is not None:
            biases = torch.chunk(merged_config.bias, split_factor, dim=0)
            for i, config in enumerate(configs):
                config.bias = biases[i]

        if merged_config.linear_type == LINEAR_COLUMN:
            if (
                merged_config.weights_scaling_factor is not None
                and merged_config.weights_scaling_factor.numel() != 1
            ):
                # INT4 AWQ and INT8 sq all linear cases
                weights_scaling_factors = torch.chunk(
                    merged_config.weights_scaling_factor, split_factor, dim=0
                )
                for i, config in enumerate(configs):
                    config.weights_scaling_factor = weights_scaling_factors[i]
        else:
            if (
                merged_config.weights_scaling_factor is not None
                and merged_config.awq_block_size != 0
            ):
                weights_scaling_factors = torch.chunk(
                    merged_config.weights_scaling_factor, split_factor, dim=1
                )
                for i, config in enumerate(configs):
                    config.weights_scaling_factor = weights_scaling_factors[i]
            if merged_config.prequant_scaling_factor is not None:
                prequant_scaling_factors = torch.chunk(
                    merged_config.prequant_scaling_factor, split_factor, dim=0
                )
                for i, config in enumerate(configs):
                    config.prequant_scaling_factor = prequant_scaling_factors[i]

    elif is_dataclass(merged_config):
        for field in fields(merged_config):
            field_configs = _split_model_config_for_tp(
                getattr(merged_config, field.name),
                split_factor,
            )
            for r in range(split_factor):
                setattr(configs[r], field.name, field_configs[r])
    elif isinstance(merged_config, list):
        for i in range(len(merged_config)):
            field_configs = _split_model_config_for_tp(
                merged_config[i],
                split_factor,
            )
            for r in range(split_factor):
                configs[r][i] = field_configs[r]

    return configs


def _split_model_config_for_pp(merged_config, split_factor):
    """This method splits ModelConfig for inference pipeline parallel."""
    num_layers = len(merged_config.layers)
    assert num_layers % split_factor == 0
    layers_per_pp = num_layers // split_factor

    configs = [copy.copy(merged_config) for _ in range(split_factor)]
    for i, config in enumerate(configs):
        if i > 0:
            config.vocab_embedding = None
            config.positional_embedding = None
        if i < split_factor - 1:
            config.ln_f = None
            config.lm_head = None
        config.layers = config.layers[i * layers_per_pp : (i + 1) * layers_per_pp]

    return configs


def _merge_model_configs_to_first_tp(config, ranks: List[int], group=None):
    """This method merges the tensor fields for linear config so the config can be used with fewer GPUs.

    The implementation is recursive.
    """
    if isinstance(config, EmbeddingConfig):
        assert config.weight is not None
        with get_tensors_parallel(config.weight, ranks, group) as weights:
            if weights:
                config.weight = torch.cat(weights, dim=0)

    elif isinstance(config, LinearConfig):
        # The scaling factors merge rule is summarized as below:

        # S: all ranks should have the same scaling factor.
        # M: Pick elementwise max among the ranks. Merged shape same as single rank.
        # C0: Concat the scaling factors on dim 0. Merged shape == tensor_parallel * original shape.
        # C1: Concat the scaling factors on dim 1. Merged shape == original shape * tensor_parallel.
        # NA: Not valid / present

        # ws: weight scaling factor
        # as: activation scaling factor
        # ps: prequant scaling factor

        # C: Colum Linear
        # R: Row Linear

        # F: FP8
        # I8: INT8 SQ
        # I4: INT4 AWQ

        # Merge Rules:
        #      ws  as  ps
        # FC   M   M   NA
        # FR   M   M   NA
        # I8C  C0  M   S
        # I8R  M   M   C0
        # I4C  C0  M   S
        # I4R  C1  M   C0

        # Handling constants
        for field_name in [
            "activation_scaling_factor",
            "weights_scaling_factor",
            "weights_scaling_factor_2",
        ]:
            field_value = getattr(config, field_name)
            if field_value is not None and field_value.numel() == 1:
                with get_tensors_parallel(field_value, ranks, group) as scaling_factors:
                    if scaling_factors:
                        # Scaling factor is a scalar.
                        setattr(
                            config,
                            field_name,
                            torch.stack(scaling_factors).max(dim=0).values,
                        )

        # We merge column linear on the dim 0 and row on the dim 1
        merge_axis = 0 if config.linear_type == LINEAR_COLUMN else 1

        assert config.weight is not None
        with get_tensors_parallel(config.weight, ranks, group) as weights:
            if weights:
                config.weight = torch.cat(weights, dim=merge_axis)

        # Only cat the bias for column linear.
        if config.linear_type == LINEAR_COLUMN and config.bias is not None:
            with get_tensors_parallel(config.bias, ranks, group) as biases:
                if biases:
                    config.bias = torch.cat(biases, dim=0)

        if config.linear_type == LINEAR_COLUMN:
            if (
                config.weights_scaling_factor is not None
                and config.weights_scaling_factor.numel() != 1
            ):
                # INT8 sq
                with get_tensors_parallel(
                    config.weights_scaling_factor, ranks, group
                ) as w_scaling_factors:
                    if w_scaling_factors:
                        config.weights_scaling_factor = torch.cat(w_scaling_factors, dim=0)
            if config.prequant_scaling_factor is not None:
                with get_tensors_parallel(
                    config.prequant_scaling_factor, ranks, group
                ) as p_scaling_factors:
                    if p_scaling_factors:
                        # INT4 AWQ, desmooth and de-smooth and re-smooth across all ranks
                        if config.awq_block_size != 0:
                            (
                                config.weight,
                                config.weights_scaling_factor,
                                config.prequant_scaling_factor,
                            ) = re_smooth_and_get_scale(
                                config.weight, p_scaling_factors, len(ranks), config.awq_block_size
                            )
                        else:
                            assert _same_tensor(
                                p_scaling_factors
                            ), f"Failed to merge config {config} with others"
        else:
            if config.weights_scaling_factor is not None:
                with get_tensors_parallel(
                    config.weights_scaling_factor, ranks, group
                ) as w_scaling_factors:
                    if w_scaling_factors:
                        if config.awq_block_size != 0:
                            # INT4 AWQ
                            if w_scaling_factors[0].ndim == 2:
                                scaling_factors_total_size = 0
                                for _, w_scaling_factor in enumerate(w_scaling_factors):
                                    scaling_factors_total_size += w_scaling_factor.numel()
                                if scaling_factors_total_size != config.weight.numel():
                                    # The weights from each rank are padded to a multiple of group_size in this case.
                                    # We need to merge the weights and recalculate the scaling factors.
                                    config.weights_scaling_factor = get_weights_scaling_factor(
                                        config.weight, config.awq_block_size
                                    )
                                else:
                                    config.weights_scaling_factor = torch.cat(
                                        w_scaling_factors, dim=1
                                    )
                            else:
                                raise NotImplementedError(
                                    "Unexpected dimensions for scaling factors."
                                )
                        else:
                            # INT8 SQ
                            config.weights_scaling_factor = (
                                torch.stack(w_scaling_factors).max(dim=0).values
                            )
            if config.prequant_scaling_factor is not None:
                with get_tensors_parallel(
                    config.prequant_scaling_factor, ranks, group
                ) as p_scaling_factors:
                    if p_scaling_factors:
                        config.prequant_scaling_factor = torch.cat(p_scaling_factors, dim=0)

    elif is_dataclass(config):
        for field in fields(config):
            _merge_model_configs_to_first_tp(getattr(config, field.name), ranks, group)
    elif isinstance(config, list):
        for i in range(len(config)):
            _merge_model_configs_to_first_tp(config[i], ranks, group)


def _model_model_configs_to_first_pp(model_config: ModelConfig, ranks: List[int]):
    """Merges the mode_config from each rank to the first pp rank."""
    # TODO: There is an NCCL error if we try group sync based on the pp_ranks.
    # So we just ask all groups to sync together for now.
    group = None

    # Merge decoder layers.
    decoder_layers = []
    for layer in model_config.layers:
        with get_configs_parallel(layer, ranks, group) as layer_configs:
            if layer_configs:
                decoder_layers.append(layer_configs[0])
                for config in layer_configs[1:]:
                    if config:
                        # Have to copy the config from the other pps as the shm will be releases after
                        decoder_layers.append(copy.deepcopy(config))

    model_config.layers = decoder_layers

    # Get the final_layernorm from the last PP rank
    with get_configs_parallel(model_config.final_layernorm, ranks, group) as configs:
        if configs and configs[-1] is not None:
            model_config.final_layernorm = configs

    # Get the lm_head from the last PP rank
    with get_configs_parallel(model_config.lm_head, ranks, group) as configs:
        if configs and configs[-1] is not None:
            model_config.lm_head = configs


def postprocess_model_config(
    model_config,
    inference_tensor_parallel: int = 1,
    inference_pipeline_parallel: int = 1,
    training_pipeline_parallel: int = 1,
) -> List[ModelConfig]:
    """Postprocesses the model configs with trained tensor parallel to target inference tensor parallel.

    If the training_pipeline_parallel > 1, the model configs across PP will be merged to one.

    Returns:
        The processed model config as a list.
            For the merging case:
                The merged rank will return the merged model_config as an single item list.
                The other ranks will return an empty list as we no longer export them.
            For the split case:
                The splitted model config list is returned.
    """
    rank = get_rank()

    # We assume the ranks ardistributed in training as [PP size, TP size].
    training_tensor_parallel = get_world_size() // training_pipeline_parallel
    tp_rank = rank % training_tensor_parallel
    pp_rank = rank // training_tensor_parallel

    print(f"current rank: {rank}, tp rank: {tp_rank}, pp rank: {pp_rank}")

    # Merge PP ranks to the first
    if training_pipeline_parallel > 1:
        # The pp_ranks for the same tp is [tp_rank, tp_rank + pp, tp_rank + pp * 2, ...]
        pp_ranks = torch.arange(
            tp_rank, get_world_size(), training_pipeline_parallel, dtype=int
        ).tolist()

        print(f"PP: Current rank {rank}, merge to {pp_ranks[0]}. Merge group {pp_ranks}")
        _model_model_configs_to_first_pp(
            model_config,
            pp_ranks,
        )

    # Returns the empty model_config on other PP ranks.
    if pp_rank != 0:
        model_config.rank = -1
        return []

    tp_world_size = get_world_size() // training_pipeline_parallel
    if inference_tensor_parallel < tp_world_size:
        # Merge the model_configs to target inference tensor parallel.
        assert (
            tp_world_size % inference_tensor_parallel == 0
        ), f"Cannot merge {tp_world_size} configs to {inference_tensor_parallel}"

        num_configs_per_group = tp_world_size // inference_tensor_parallel
        local_tp_group_id = tp_rank // num_configs_per_group
        tp_ranks = list(
            range(
                local_tp_group_id * num_configs_per_group,
                (local_tp_group_id + 1) * num_configs_per_group,
            )
        )

        print(f"TP: Current rank {rank}, merge to {tp_ranks[0]}. Merge group {tp_ranks}.")
        # We sync on all TP ranks (and pp_rank = 0)
        group = get_group(list(range(training_tensor_parallel)))
        _merge_model_configs_to_first_tp(model_config, tp_ranks, group)
        model_config.tensor_parallel = inference_tensor_parallel
        if rank == tp_ranks[0]:
            model_config.rank = local_tp_group_id
            splitted_model_configs = [model_config]
        else:
            # Mark this config to be invalid and return it as invalid.
            model_config.rank = -1
            return []

    elif inference_tensor_parallel > tp_world_size:
        assert (
            tp_world_size == 1
        ), "We only support splitting a single model config to multiple GPUs"
        split_factor = inference_tensor_parallel // tp_world_size
        splitted_model_configs = _split_model_config_for_tp(
            model_config,
            split_factor,
        )
        for i, config in enumerate(splitted_model_configs):
            config.rank = i
            config.tensor_parallel = inference_tensor_parallel

    else:
        splitted_model_configs = [model_config]

    if inference_pipeline_parallel > 1:
        splitted_model_configs_tp_pp = []
        for i, model_config_tp in enumerate(splitted_model_configs):
            splitted_model_configs_pp = _split_model_config_for_pp(
                model_config_tp, inference_pipeline_parallel
            )
            for j, config in enumerate(splitted_model_configs_pp):
                config.rank = i + j * inference_tensor_parallel
                config.tensor_parallel = inference_tensor_parallel
                config.pipeline_parallel = inference_pipeline_parallel
            splitted_model_configs_tp_pp.extend(splitted_model_configs_pp)
        return splitted_model_configs_tp_pp
    else:
        return splitted_model_configs


def pad_embedding_lm_head(model_config: ModelConfig, padding_factor: int = 64):
    """Pad lm_head and embedding as multiples of 64 for AWQ quantization."""
    vocab_size = model_config.vocab_size
    # Pad the lm_head and vocab_embedding only if the lm_head is quantized with AWQ.
    if vocab_size % padding_factor == 0:
        return

    pad_vocab_size = int((vocab_size + padding_factor - 1) / padding_factor) * padding_factor
    model_config.vocab_size = pad_vocab_size

    if hasattr(model_config, "vocab_embedding"):
        embedding_config = model_config.vocab_embedding
        original_weight = embedding_config.weight
        pad_size = (0, 0, 0, pad_vocab_size - original_weight.shape[0])
        embedding_config.weight = torch.nn.functional.pad(
            original_weight, pad_size, mode="constant", value=0
        )

    if hasattr(model_config, "lm_head"):
        lm_head_config = model_config.lm_head
        original_weight = lm_head_config.weight
        original_bias = lm_head_config.bias
        original_wsf = lm_head_config.weights_scaling_factor

        lm_head_config.weight = torch.nn.functional.pad(
            original_weight,
            (0, 0, 0, pad_vocab_size - original_weight.shape[0]),
            mode="constant",
            value=0,
        )
        if original_bias is not None:
            lm_head_config.bias = torch.nn.functional.pad(
                original_bias,
                (0, pad_vocab_size - original_bias.shape[0]),
                mode="constant",
                value=0,
            )

        if original_wsf is not None:
            assert len(original_wsf.shape) == 2, "AWQ weight scaling factor should be 2D."
            pad_weights_scaling_factor = (
                torch.ones(
                    (pad_vocab_size, original_wsf.shape[1]),
                    dtype=original_wsf.dtype,
                )
                / 7.0  # int4: maxbound = 7.0
            )

            pad_weights_scaling_factor[:vocab_size, :] = original_wsf
            lm_head_config.weights_scaling_factor = pad_weights_scaling_factor


def check_weight_shape_valid(config, inference_tensor_parallel=1, training_tensor_parallel=1):
    """Check if weight shape are valid with inference TP.

    This function is recurisve.
    """
    if isinstance(config, LinearConfig):
        # check weight shape
        if config.linear_type == LINEAR_COLUMN:
            _, k = config.weight.shape
        else:
            k, _ = config.weight.shape
        merged_k = k * training_tensor_parallel
        assert (
            merged_k % inference_tensor_parallel == 0
        ), f"Weights cannot be split into {inference_tensor_parallel} ranks."
        if (
            config.awq_block_size > 0
            and (merged_k // inference_tensor_parallel) % config.awq_block_size != 0
        ):
            raise NotImplementedError(
                "Weight shape is not divisible for block size for block quantization."
            )
        return

    if is_dataclass(config):
        for field in fields(config):
            check_weight_shape_valid(
                getattr(config, field.name),
                inference_tensor_parallel,
                training_tensor_parallel,
            )
    elif isinstance(config, list):
        for config_i in config:
            check_weight_shape_valid(config_i, inference_tensor_parallel, training_tensor_parallel)


def postprocess_tensors(
    model_config: ModelConfig,
    force_cpu: bool = True,
    force_contiguous: bool = True,
    force_non_view: bool = True,
):
    """Make all tensors in the model_config are on CPU, contiguous and own the memory."""

    def _postprocess_tensor(tensor):
        if force_cpu:
            tensor = tensor.cpu()
        if force_contiguous:
            tensor = tensor.contiguous()
        if force_non_view and tensor._is_view():
            tensor = tensor.clone()
        return tensor

    for field in fields(model_config):
        field_value = getattr(model_config, field.name)

        if isinstance(field_value, torch.Tensor):
            setattr(model_config, field.name, _postprocess_tensor(field_value))
        elif isinstance(field_value, list):
            for i, v in enumerate(field_value):
                if isinstance(v, torch.Tensor):
                    field_value[i] = _postprocess_tensor(v)
                elif is_dataclass(v):
                    postprocess_tensors(v)
        elif is_dataclass(field_value):
            postprocess_tensors(field_value)
