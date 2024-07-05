# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Defines the tensorrt_llm inference API that can support both single and multiple GPU LLM inferences.

Referrence impl in tensorrt_llm: examples/llama/summarize.py.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorrt as trt
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import PreTrainedTokenizer

from .tensorrt_llm_build import find_engines  # isort:skip


@dataclass
class TensorrtLLMHostContext:
    """The host side context for TRT LLM inference."""

    executor: MPIPoolExecutor = None
    tensor_parallel: int = 1
    tokenizer: PreTrainedTokenizer = None
    max_batch_size: int = 0
    max_input_len: int = 0


@dataclass
class TensorrtLLMWorkerContext:
    """The MPI worker side context for TRT LLM inference."""

    decoder: tensorrt_llm.runtime.GenerationSession = None
    sampling_config: SamplingConfig = None
    max_batch_size: int = 0
    max_input_len: int = 0
    num_beams: int = 1


# This is a global context that will be initialized during the model loading process as MPI worker.
tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def read_config(config_path: Path) -> Tuple[ModelConfig, int, str, int, int]:
    """Reads the `ModelConfig` from the config_path.

    Returns:
        model_config: The `ModelConfig` loaded.
        world_size: The world size, or TP.
        dtype: The default data type of this engine.
        max_input_len: Max input token lenghth.
        max_batch_size: Max batch size.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
    remove_input_padding = config["plugin_config"]["remove_input_padding"]
    tp_size = config["builder_config"]["tensor_parallel"]
    pp_size = config["builder_config"].get("pipeline_parallel", 1)
    world_size = tp_size * pp_size
    assert pp_size == 1, "pipeline_parallel is not supported"
    assert (
        world_size == tensorrt_llm.mpi_world_size()
    ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"

    assert world_size <= torch.cuda.device_count(), f"Not enough GPUs, requesting {world_size}"

    num_heads = config["builder_config"]["num_heads"]

    assert (num_heads % tp_size) == 0, "The number of heads must be a multiple of tp_size"

    num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)
    hidden_size = config["builder_config"]["hidden_size"]
    vocab_size = config["builder_config"]["vocab_size"]
    num_layers = config["builder_config"]["num_layers"]
    paged_kv_cache = config["plugin_config"]["paged_kv_cache"]

    if paged_kv_cache:
        print(
            "Warning: paged_kv_cache is designed for the C++ runtime. You may hit OOM with python."
        )

    tokens_per_block = config["plugin_config"]["tokens_per_block"]
    max_prompt_embedding_table_size = config["builder_config"].get(
        "max_prompt_embedding_table_size", 0
    )
    quant_mode = QuantMode(config["builder_config"].get("quant_mode", 0))
    dtype = config["builder_config"]["precision"]
    gather_all_token_logits = config["builder_config"].get("gather_all_token_logits", False)
    use_custom_all_reduce = config["plugin_config"].get("use_custom_all_reduce", False)
    lora_plugin = config["plugin_config"].get("lora_plugin", False)
    assert not lora_plugin, "lora_plugin is not supported"

    hidden_size = hidden_size // tp_size
    num_heads = num_heads // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        dtype=dtype,
        quant_mode=quant_mode,
        gather_all_token_logits=gather_all_token_logits,
        use_custom_all_reduce=use_custom_all_reduce,
        lora_plugin=lora_plugin,
    )

    dtype = config["builder_config"]["precision"]
    max_input_len = config["builder_config"]["max_input_len"]
    max_batch_size = config["builder_config"]["max_batch_size"]

    return model_config, world_size, dtype, max_input_len, max_batch_size


def _load(engine_dir: Union[str, Path], eos_token_id: int, num_beams=1):
    """The impl of `load` API for on a single GPU worker."""
    try:
        tensorrt_llm.logger.set_level("warning")

        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        model_config, world_size, dtype, max_input_len, max_batch_size = read_config(config_path)

        runtime_rank = tensorrt_llm.mpi_rank()

        assert runtime_rank < torch.cuda.device_count(), f"Rank {runtime_rank} out of bound"

        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank, tp_size=world_size)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        serialize_path = find_engines(
            engine_dir, dtype=dtype, tp_size=str(world_size), rank=runtime_rank
        )[0]

        with open(serialize_path, "rb") as f:
            engine_buffer = f.read()
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=False
        )

        sampling_config = SamplingConfig(
            end_id=eos_token_id, pad_id=eos_token_id, num_beams=num_beams
        )

        # Initialize the global context so it can be used during `run` API.
        global tensorrt_llm_worker_context
        tensorrt_llm_worker_context.decoder = decoder
        tensorrt_llm_worker_context.sampling_config = sampling_config
        tensorrt_llm_worker_context.max_batch_size = max_batch_size
        tensorrt_llm_worker_context.max_input_len = max_input_len
        tensorrt_llm_worker_context.num_beams = num_beams

    except Exception as e:
        print(e)
        raise e


def _forward(
    input_tensors: List[torch.IntTensor],
    max_output_len: int,
    max_attention_window_size: Optional[int] = None,
    profiler_output_path: Union[Path, str] = None,
) -> Optional[torch.IntTensor]:
    """The impl of `forward` API for on a single GPU worker with tensor as IO.

    Returns:
        the output tokens tensor with shape [batch_size, num_beams, output_len].
    """
    try:
        # Loading the global context initialized from the `load` API.
        global tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        assert decoder is not None, "Invalid worker context, decoder is not loaded."
        sampling_config = tensorrt_llm_worker_context.sampling_config
        max_batch_size = tensorrt_llm_worker_context.max_batch_size
        max_input_len = tensorrt_llm_worker_context.max_input_len
        num_beams = tensorrt_llm_worker_context.num_beams

        batch_size = len(input_tensors)
        assert (
            batch_size <= max_batch_size
        ), f"batch size {batch_size} exceedng max batch size {max_batch_size}"
        input_lengths = [t.shape[0] for t in input_tensors]
        max_length = max(input_lengths)
        assert (
            max_length <= max_input_len
        ), f"input length {max_length} exceedng max input length {max_input_len}"
        pad_id = sampling_config.pad_id

        if decoder.remove_input_padding:
            line_encoded = [t.int().cuda() for t in input_tensors]
        else:
            line_encoded = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tensors, dtype=torch.int32), pad_id
            ).cuda()
            input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()

        class _TRTLLMProfiler(trt.IProfiler):
            def __init__(self, output):
                super().__init__()
                self.output = output

            def report_layer_time(self, layer_name, ms):
                self.output.append([layer_name, ms])

        with torch.no_grad():
            ctx_context_profiler = []
            context_0_profiler = []
            context_1_profiler = []
            if profiler_output_path:
                decoder.runtime.ctx_context.profiler = _TRTLLMProfiler(ctx_context_profiler)
                decoder.runtime.context_0.profiler = _TRTLLMProfiler(context_0_profiler)
                decoder.runtime.context_1.profiler = _TRTLLMProfiler(context_1_profiler)

            decoder.setup(
                batch_size,
                max_context_length=max_length,
                max_new_tokens=max_output_len,
                beam_width=num_beams,
                max_attention_window_size=max_attention_window_size,
            )

            if decoder.remove_input_padding:
                output_ids = decoder.decode_batch(line_encoded, sampling_config)
            else:
                output_ids = decoder.decode(
                    line_encoded,
                    input_lengths,
                    sampling_config,
                )

            torch.cuda.synchronize()

            runtime_rank = tensorrt_llm.mpi_rank()
            if profiler_output_path:
                if len(ctx_context_profiler) == 0 or len(context_0_profiler) == 0:
                    print(
                        "\nWarning: TRT profiler overflow. The profiling results are truncated."
                        " Please reduce max_output_len and retry the profiling.\n"
                    )
                with open(profiler_output_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    for context_name, output in zip(
                        ["ctx_context", "context_0", "context_1"],
                        [ctx_context_profiler, context_0_profiler, context_1_profiler],
                    ):
                        for entry in output:
                            writer.writerow([context_name, runtime_rank, entry[0], entry[1]])

            if runtime_rank == 0:
                return output_ids
            else:
                return None

    except Exception as e:
        print(e)
        raise e


def load(
    tokenizer: PreTrainedTokenizer, engine_dir: Union[str, Path], num_beams: int = 1
) -> TensorrtLLMHostContext:
    """Loaded the compiled LLM model and run it.

    It also supports running the TRT LLM model on multi-GPU.
    """
    engine_dir = Path(engine_dir)
    config_path = engine_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    tensor_parallel = config["builder_config"]["tensor_parallel"]
    if tensor_parallel == 1:
        _load(engine_dir, tokenizer.eos_token_id, num_beams)
        executor = None
    else:
        executor = MPIPoolExecutor(max_workers=tensor_parallel)
        futures = []
        for _ in range(tensor_parallel):
            future = executor.submit(_load, engine_dir, tokenizer.eos_token_id, num_beams)
            futures.append(future)
        for future in futures:
            future.result()

    max_batch_size = config["builder_config"]["max_batch_size"]
    max_input_len = config["builder_config"]["max_input_len"]

    return TensorrtLLMHostContext(
        executor=executor,
        tensor_parallel=tensor_parallel,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
    )


def unload(host_context: TensorrtLLMHostContext):
    """Frees the GPU resource from the TensorrtLLMHostContext and reset the host_context."""
    if host_context.executor is not None:
        host_context.executor.shutdown(wait=True)
        host_context.executor = None
        return

    global tensorrt_llm_worker_context
    tensorrt_llm_worker_context.decoder = None
    tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def forward(
    input_tensors: List[torch.IntTensor],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    max_attention_window_size: Optional[int] = None,
    profiler_output_path: Union[Path, str] = None,
) -> Optional[torch.IntTensor]:
    """Run the loaded model with the host_context provided from the `load` API."""
    batch_size = len(input_tensors)
    max_batch_size = host_context.max_batch_size
    assert (
        batch_size <= max_batch_size
    ), f"batch size {batch_size} exceedng max batch size {max_batch_size}"
    max_length = max([t.shape[0] for t in input_tensors])
    max_input_len = host_context.max_input_len
    assert (
        max_length <= max_input_len
    ), f"input length {max_length} exceedng max input length {max_input_len}"

    if profiler_output_path:
        print(f"Dumping TRT profiler result to {profiler_output_path} as CSV")
        with open(profiler_output_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["context", "rank", "layer", "latency_ms"])

    tensor_parallel = host_context.tensor_parallel
    if tensor_parallel == 1:
        return _forward(
            input_tensors, max_output_len, max_attention_window_size, profiler_output_path
        )
    else:
        executor = host_context.executor
        futures = []
        for _ in range(tensor_parallel):
            future = executor.submit(
                _forward,
                input_tensors,
                max_output_len,
                max_attention_window_size,
                profiler_output_path,
            )
            futures.append(future)
        for future in futures:
            result = future.result()
            if result is not None:
                return result

        raise RuntimeError("Internal error")


def generate(
    input_texts: List[str],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    max_attention_window_size: Optional[int] = None,
    profiler_output_path: Union[Path, str] = None,
) -> List[List[str]]:
    """Generate the output sequences from the input sequences.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer
    max_input_len = host_context.max_input_len
    input_tensors = [
        torch.IntTensor(
            tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=max_input_len)
        )
        for t in input_texts
    ]
    output_tensor = forward(
        input_tensors, max_output_len, host_context, max_attention_window_size, profiler_output_path
    )
    assert output_tensor is not None

    input_lengths = [t.shape[0] for t in input_tensors]
    output_lines_list = [
        tokenizer.batch_decode(output_tensor[b, :, input_lengths[b] :], skip_special_tokens=True)
        for b in range(output_tensor.shape[0])
    ]
    return output_lines_list
