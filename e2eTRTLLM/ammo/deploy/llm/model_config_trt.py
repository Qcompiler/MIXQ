# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The API convert the model_config format to tensorrt_llm."""

import copy
import json
import multiprocessing
import os
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Iterator, List, Optional, Union

import numpy as np
import psutil
import torch
from safetensors import safe_open

from ammo.torch.export import (
    AWQ_SUPPORTED_VERSION,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_W4A8_AWQ,
    SUPPORTED_VERSION,
    ModelConfig,
    model_config_from_dict,
    postprocess_tensors,
    restore_model_config,
)

from .tensorrt_llm_model import LMHeadModelBuilder

gpu_process_list: List[Optional[Process]] = [None] * torch.cuda.device_count()


def load_model_configs(model_config_json: Union[str, Path]) -> Iterator[ModelConfig]:
    """Loads the model_config saved from AMMO export.

    Args:
        model_config_json: The exported json file from AMMO describing the optimized model.
            Inside the same directory, each gpu rank will have its own safetensors/npz file.
            The json file represents the general ModelConfig structure while the detailed
            weights for each rank are stored in the safetensors/npz file.

    Returns:
        The list of `ModelConfig` loaded and constructed.
    """
    model_config_json = Path(model_config_json)
    assert model_config_json.exists()

    with open(model_config_json, "r") as f:
        model_config_template = json.load(f)

    tensor_parallel = model_config_template["tensor_parallel"]
    assert tensor_parallel > 0, f"Invalid tensor_parallel {tensor_parallel}"

    model_config_dir = model_config_json.parents[0]

    for i in range(tensor_parallel):
        weights = None

        # Old export format: weights are stored as an NPZ file.
        weights_file_npz = f"*rank{i}.npz"
        npz_paths = list(model_config_dir.glob(weights_file_npz))
        if npz_paths:
            weights = dict(np.load(npz_paths[0]))
            print(
                "Warning: the npz checkpoints is deprecated. To ensure future TensorRT-LLM engine"
                " compatibility, please re-export the quantized model."
            )

        # New export format: weights are stored as a safetensors file.
        weights_file_safetensors = f"rank{i}.safetensors"
        safetensors_path = model_config_dir / weights_file_safetensors
        if safetensors_path.exists():
            weights = {}
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)

        assert weights is not None, f"Cannot load weights for rank{i}"

        model_config = copy.deepcopy(model_config_template)
        model_config["rank"] = i
        restore_model_config(model_config, weights)
        loaded_model_config = model_config_from_dict(model_config)
        assert loaded_model_config.version >= SUPPORTED_VERSION, (
            f"Model config {model_config_json} version is not supported. Please use the latest AMMO"
            " release and export the optimized model config again."
        )

        assert loaded_model_config.pipeline_parallel == 1, (
            "ammo.deploy.llm does not support pipeline_parallel. Please use TensorRT-LLM build API"
            " to build the engines"
        )

        if loaded_model_config.quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
            assert loaded_model_config.version >= AWQ_SUPPORTED_VERSION, (
                f"Model config {model_config_json} version is not supported. Please use the latest"
                " AMMO release and export the optimized model config again."
            )

        yield loaded_model_config


def _model_config_to_tensorrt_llm_impl(
    model_config: ModelConfig,
    engine_dir: Path,
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    inflight_batching: bool = False,
    enable_sparsity: bool = False,
    refit_engine_path: Path = "",
):
    builder = LMHeadModelBuilder(model_config)
    builder.build(
        output_dir=engine_dir,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        inflight_batching=inflight_batching,
        enable_sparsity=enable_sparsity,
        refit_engine_path=refit_engine_path,
    )


def model_config_to_tensorrt_llm(
    model_config: ModelConfig,
    engine_dir: Union[str, Path],
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    num_build_workers: int = 1,
    inflight_batching: bool = False,
    enable_sparsity: bool = False,
    refit_engine_dir: Union[str, Path] = None,
):
    """The API to convert the ModelConfig to tensorrt_llm for a single GPU rank.

    Args:
        model_config: The ModelConfig converted, for a single GPU rank.
        engine_dir: The target output directory to save the built tensorrt_llm engines.
        max_input_len: The max input sequence length.
        max_output_len: The max output sequence length.
        max_batch_size: The max batch size.
        max_beam_width: The max beam search width.
        num_build_workers: The number of workers to use for the building process.
            If build time is a concern, you can increase this worker count to num of GPUs.
            At a lost of higer CPU memory usage footprint.
            If CPU memory is limited, num_build_workers should be set to 1 to conserve memory.
        inflight_batching: Whether to build the engine to support inflight_batching.
            Engines build with inflight_batching enabled is designed to run with the C++ runtime.
            It can perform worse in the python runtime.
        enable_sparsity: The switch to enable sparsity for TRT compiler.
            With this flag, the TRT compiler will search tactics of sparse kernels for each node of which
            weight tensors are sparsified. This increases engine building time significantly.
        refit_engine_dir: If provided, the built engine will be a refitting of the previously built engine(s).
            User has to validate that the engines under refit_engine_dir are built with the same model networks
            and configs.
    """
    if inflight_batching:
        print(
            "Engines build with inflight_batching enabled is designed to run with the C++ runtime."
            " It can perform worse in the python runtime."
        )

    engine_dir = Path(engine_dir)
    if refit_engine_dir:
        refit_engine_dir = Path(refit_engine_dir)
        if not refit_engine_dir.exists():
            print(f"{refit_engine_dir} does not exist. Building from scratch.")
            refit_engine_dir = None
    if model_config.rank == 0 and os.path.exists(engine_dir) and refit_engine_dir != engine_dir:
        for engine_file in engine_dir.glob("*.engine"):
            print(f"Removing previous engine file: {engine_file}")
            os.remove(engine_file)

    print(
        f"Before engine building rank {model_config.rank}, CPU RAM Used (GB):"
        f" {psutil.Process().memory_info().rss / 1024 / 1024 / 1024}"
    )

    # Build model in a separate process so the extra memory used in the TRT-LLM building stage
    # can be recycled after building finished.
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    global gpu_process_list

    num_gpus = min(torch.cuda.device_count(), num_build_workers)
    gpu_id = model_config.rank % num_gpus

    if gpu_process_list[gpu_id] is not None:
        gpu_process_list[gpu_id].join()
        gpu_process_list[gpu_id] = None

    blocking_build = gpu_id == num_gpus - 1 or model_config.rank == model_config.tensor_parallel - 1

    refit_engine_path = None
    if refit_engine_dir:
        refit_engine_path = (
            engine_dir
            / f"ammo_{model_config.dtype}_tp{model_config.tensor_parallel}_rank{model_config.rank}.engine"
        )
        assert refit_engine_path.exists(), f"Cannot find refit engine file {refit_engine_path}"
        print(f"Refitting rank {model_config.rank} with {refit_engine_path}")

    # Use Thread to launch the build process in the current process if not building in paralllel.
    launch_way = Thread if blocking_build else Process

    if launch_way == Process:
        # We cannot send a tensor view to the sub process.
        postprocess_tensors(
            model_config, force_cpu=True, force_contiguous=True, force_non_view=True
        )

    p = launch_way(
        target=_model_config_to_tensorrt_llm_impl,
        args=(
            model_config,
            engine_dir,
            max_input_len,
            max_output_len,
            max_batch_size,
            max_beam_width,
            inflight_batching,
            enable_sparsity,
            refit_engine_path,
        ),
    )
    gpu_process_list[gpu_id] = p
    p.start()

    # If we used all the gpus or we are building the last rank, we drain the engine build process list.
    if blocking_build:
        for i, p in enumerate(gpu_process_list):
            if p is not None:
                p.join()
                gpu_process_list[i] = None

        print(
            f"After Engine building rank {model_config.rank}, CPU RAM Used (GB):"
            f" {psutil.Process().memory_info().rss / 1024 / 1024 / 1024}"
        )
