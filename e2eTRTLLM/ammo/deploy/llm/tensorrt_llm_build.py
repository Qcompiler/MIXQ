# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This module builds the tensorrt_llm engine.

Referrence impl in tensorrt_llm: examples/gpt/build.py.
"""

import argparse
import math
import os
import shutil
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.profiler import check_gpt_mem_usage
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "ammo"


def get_engine_name(model, dtype, tp_size, rank):
    """Returns the engine file name based on the provided info."""
    return "{}_{}_tp{}_rank{}.engine".format(model, dtype, tp_size, rank)


def find_engines(
    dir: Path, model_name: str = "*", dtype: str = "*", tp_size: str = "*", rank: str = "*"
) -> List[Path]:
    """Globs the engine file path under dir."""
    template = f"{model_name}_{dtype}_tp{tp_size}_rank{rank}.engine"
    return list(dir.glob(template))


def serialize_engine(engine, path):
    """Serializes the engine to path."""
    logger.info(f"Serializing engine to {path}...")
    tik = time.time()
    with open(path, "wb") as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Engine serialized. Total time: {t}")


def build_rank_engine(
    tensorrt_llm_gpt,
    builder: Builder,
    builder_config: tensorrt_llm.builder.BuilderConfig,
    engine_name,
    rank,
    args,
):
    """@brief: Build the engine on the given rank.

    @param rank: The rank to build the engine.
    @param args: The cmd line arguments.
    @return: The built engine.
    """
    str_dtype_to_trt(args.dtype)

    # TODO: Enable use_embedding_sharing when this feature is needed.
    # Share_embedding_table can be set True only when:
    # 1) the weight for lm_head() does not exist while other weights exist
    # 2) For multiple-processes, use_parallel_embedding=True and embedding_sharding_dim == 0.
    # Besides, for TensorRT 9.0, we can observe the engine size reduction when the lookup and gemm plugin are enabled.
    # share_embedding_table = False
    # if args.use_embedding_sharing:
    #     if args.world_size > 1:
    #         if args.model_dir is not None and args.embedding_sharding_dim == 0 and args.use_parallel_embedding:
    #             share_embedding_table = check_embedding_share(args.model_dir)
    #     else:
    #         if args.model_dir is not None:
    #             share_embedding_table = check_embedding_share(args.model_dir)

    #     if not share_embedding_table:
    #         logger.warning(f'Cannot share the embedding lookup table.')

    # if share_embedding_table:
    #     logger.info(
    #         'Engine will share embedding and language modeling weights.')

    # Module -> Network
    ootb = os.getenv("OOTB", False)

    network = builder.create_network()
    network.trt_network.name = engine_name

    # We have to use the attention plugin for most of the models.
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(dtype=args.use_gpt_attention_plugin)

    # AWQ
    if tensorrt_llm_gpt.quant_mode == QuantMode.PER_GROUP | QuantMode.INT4_WEIGHTS:
        args.use_weight_only = True
        args.per_group = True

    # Enable sparsity in builder config
    if args.enable_sparsity:
        builder_config._trt_builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        # For a module, it is likely only some of the instances are sparisified
        builder_config._trt_builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

    if not ootb:
        if args.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        if args.use_rmsnorm_plugin:
            network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)
        if args.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(dtype=args.use_layernorm_plugin)
        assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
        if args.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if args.enable_context_fmha_fp32_acc:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
        if args.multi_block_mode:
            network.plugin_config.enable_mmha_multi_block_mode()
        if args.remove_input_padding:
            network.plugin_config.enable_remove_input_padding()
        if args.paged_kv_cache:
            network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)
        if args.use_lora_plugin:
            network.plugin_config.set_lora_plugin(dtype=args.use_lora_plugin)
        if args.use_weight_only and args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(dtype="float16")

        if args.use_lookup_plugin:
            # Use the plugin for the embedding parallelism and sharing
            network.plugin_config.set_lookup_plugin(args.dtype, args.use_custom_all_reduce)

        if args.use_paged_context_fmha or args.max_draft_len > 0:
            assert (
                args.enable_context_fmha or args.enable_context_fmha_fp32_acc
            ), "context fmha must be enabled"
            network.plugin_config.set_paged_context_fmha()

        if args.use_context_fmha_for_generation:
            logger.warning(
                "use_context_fmha_for_generation is set. This flag must be used only for testing"
            )
            assert (
                args.use_gpt_attention_plugin
                and args.paged_kv_cache
                and args.use_paged_context_fmha
            ), "use_context_fmha_for_generation must be used with paged KV cache and attention."
            network.plugin_config.set_context_fmha_for_generation()

        if args.max_draft_len > 0:
            network.plugin_config.set_paged_context_fmha()
    else:
        print("Build engine in OOTB mode, disable all plugins except nccl.")

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = tensorrt_llm_gpt.prepare_inputs(
            args.max_batch_size,
            args.max_input_len,
            args.max_output_len,
            True,
            args.max_beam_width,
            args.max_num_tokens,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size,
            gather_all_token_logits=args.gather_all_token_logits,
            max_draft_len=args.max_draft_len,
            lora_target_modules=args.lora_target_modules,
        )
        tensorrt_llm_gpt(*inputs)

    # Turn on graph_rewriting by default.
    tensorrt_llm.graph_rewriting.optimize(network)

    engine = None

    # Network -> Engine
    if args.refit_engine_path:
        try:
            with open(args.refit_engine_path, "rb") as engine_file:
                engine_buffer = engine_file.read()
            engine = builder.refit_engine(network, engine_buffer)
            if rank == 0:
                source_config_path = args.refit_engine_path.parent / "config.json"
                config_path = args.output_dir / "config.json"
                shutil.copy(source_config_path, config_path)
        except Exception as e:
            print(f"Cannot refit engine with {args.refit_engine_path}, error {e}")
            print("Re-build engine from scratch")

    if not engine:
        engine = builder.build_engine(network, builder_config)
        if rank == 0:
            config_path = args.output_dir / "config.json"
            builder.save_config(builder_config, config_path)

    return engine


def _build_impl(rank, tensorrt_llm_model, args):
    device_id = rank % args.gpus_per_node
    torch.cuda.set_device(device_id)
    tensorrt_llm.logger.set_level(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(device_id).replace(" ", "_")
    timing_cache_file = (
        args.timing_cache if args.timing_cache else args.output_dir / f"model.cache.{device_name}"
    )
    print(f"Using timing cache {timing_cache_file}")
    timing_cache = timing_cache_file

    builder = Builder()
    apply_query_key_layer_scaling = False
    cur_rank = rank

    args.int8 = "int8" in args.quantization
    #  The vocab size needs to be padded for int4_awq
    if QuantMode.PER_GROUP | QuantMode.INT4_WEIGHTS == tensorrt_llm_model.quant_mode:
        assert (
            tensorrt_llm_model._vocab_size % 64 == 0
        ), "The vocab_size is not multiples of 64 for AWQ quant_mode."
        # If TRT-LLM packs int4_awq weights into int8, we need to enable int8 flags for builder
        if tensorrt_llm_model.layers[0].attention.qkv.qweight._value.dtype == np.int8:
            args.int8 = True

    int8 = args.int8
    fp8 = "fp8" in args.quantization

    # We support refit by default.
    # TRT does not support quantized model refit.
    use_refit = not int8 and not fp8
    if not use_refit and args.refit_engine_path:
        print("Warning: quantized engine cannot be used for refitting. Refitting is disabled.")
        args.refit_engine_path = ""

    builder_config = builder.create_builder_config(
        name=MODEL_NAME,
        precision=args.dtype,
        timing_cache=timing_cache,
        tensor_parallel=args.world_size,  # TP only
        num_layers=tensorrt_llm_model._num_layers,
        num_heads=tensorrt_llm_model._num_heads,
        num_kv_heads=tensorrt_llm_model._num_kv_heads,
        hidden_size=tensorrt_llm_model._hidden_size,
        vocab_size=tensorrt_llm_model._vocab_size,
        hidden_act=tensorrt_llm_model.hidden_act,
        max_position_embeddings=tensorrt_llm_model.max_position_embeddings,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_beam_width=args.max_beam_width,
        max_output_len=args.max_output_len,
        max_num_tokens=args.max_num_tokens,
        max_draft_len=args.max_draft_len,
        int8=int8,
        fp8=fp8,
        opt_level=args.builder_opt,
        strongly_typed=args.strongly_typed,
        max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
        gather_all_token_logits=args.gather_all_token_logits,
        quant_mode=tensorrt_llm_model.quant_mode,
        use_parallel_embedding=args.use_parallel_embedding,
        lora_target_modules=args.lora_target_modules,
        use_refit=use_refit,
    )

    engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size, cur_rank)
    engine = build_rank_engine(
        tensorrt_llm_model, builder, builder_config, engine_name, cur_rank, args
    )
    assert engine is not None, f"Failed to build engine for rank {cur_rank}"

    local_num_kv_heads = (
        tensorrt_llm_model._num_kv_heads + tensorrt_llm_model._tp_size - 1
    ) // tensorrt_llm_model._tp_size
    kv_dtype = str_dtype_to_trt(args.dtype)
    # TODO: set kv_dtype to int8 once we enable int8 kv cache.
    if tensorrt_llm_model.quant_mode:
        kv_dtype = str_dtype_to_trt("fp8")
    check_gpt_mem_usage(
        engine=engine,
        kv_dtype=kv_dtype,
        use_gpt_attention_plugin=args.use_gpt_attention_plugin,
        paged_kv_cache=args.paged_kv_cache,
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_beam_width,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        local_num_kv_heads=local_num_kv_heads,
        head_size=tensorrt_llm_model._hidden_size / tensorrt_llm_model._num_heads,
        num_layers=tensorrt_llm_model._num_layers,
    )

    serialize_engine(engine, args.output_dir / engine_name)
    del engine

    if rank == 0:
        ok = builder.save_timing_cache(builder_config, timing_cache_file)
        assert ok, "Failed to save timing cache."


def build(
    tensorrt_llm_model,
    output_dir: Path,
    rank=0,
    world_size=1,
    dtype="float16",
    timing_cache="",
    log_level="warning",
    max_batch_size=1,
    max_input_len=200,
    max_output_len=200,
    max_beam_width=1,
    gpus_per_node=1,
    quantization=None,
    inflight_batching=False,
    enable_sparsity=False,
    refit_engine_path="",
):
    """Builds the tensorrt_llm_model to engine."""
    args = argparse.Namespace()
    args.world_size = world_size
    args.dtype = dtype
    args.timing_cache = timing_cache
    args.log_level = log_level
    args.max_batch_size = max_batch_size
    args.max_input_len = max_input_len
    args.max_output_len = max_output_len
    args.max_beam_width = max_beam_width
    args.use_gpt_attention_plugin = dtype
    args.use_gemm_plugin = dtype
    args.use_rmsnorm_plugin = False
    args.use_layernorm_plugin = False
    args.enable_context_fmha = True
    args.enable_context_fmha_fp32_acc = False
    args.multi_block_mode = False
    args.gpus_per_node = gpus_per_node
    args.builder_opt = None
    args.output_dir = Path(output_dir)
    args.remove_input_padding = True
    args.use_smooth_quant = False
    args.use_weight_only = False
    args.weight_only_precision = "int8"
    args.per_channel = False
    args.per_token = False
    args.random_seed = None
    args.paged_kv_cache = False
    args.tokens_per_block = 128
    args.max_prompt_embedding_table_size = 0
    args.use_inflight_batching = inflight_batching
    args.use_parallel_embedding = True
    args.embedding_sharding_dim = 0
    args.use_embedding_sharing = False
    args.use_lookup_plugin = False
    args.max_num_tokens = None
    args.gather_all_token_logits = False
    args.strongly_typed = any([key in quantization for key in ["fp8", "int8"]])
    args.use_custom_all_reduce = False
    args.use_lora_plugin = False
    args.max_draft_len = 0
    args.use_paged_context_fmha = False
    args.use_context_fmha_for_generation = False
    args.lora_target_modules = None
    args.moe_num_experts = 0
    args.moe_top_k = 0
    args.moe_tp_mode = MoeConfig.ParallelismMode.TENSOR_PARALLEL
    args.moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
    args.quantization = quantization
    args.per_group = False
    args.enable_sparsity = enable_sparsity

    args.refit_engine_path = refit_engine_path

    if args.use_inflight_batching:
        print(
            "Warning: inflight batching can only be launched in the C++ runtime. Python runtime"
            " will hit GPU OOM issue."
        )

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if not args.remove_input_padding:
        if args.use_gpt_attention_plugin:
            logger.warning(
                "It is recommended to specify --remove_input_padding when using GPT attention"
                " plugin"
            )

    if args.use_inflight_batching:
        if not args.use_gpt_attention_plugin:
            args.use_gpt_attention_plugin = "float16"
            logger.info(
                "Using GPT attention plugin for inflight batching mode. Setting to default"
                f" '{args.use_gpt_attention_plugin}'"
            )
        if not args.remove_input_padding:
            args.remove_input_padding = True
            logger.info("Using remove input padding for inflight batching mode.")
        if not args.paged_kv_cache:
            args.paged_kv_cache = True
            logger.info("Using paged KV cache for inflight batching mode.")

    assert math.log2(args.tokens_per_block).is_integer(), "tokens_per_block must be power of 2"
    if args.enable_context_fmha or args.enable_context_fmha_fp32_acc:
        assert args.tokens_per_block >= 128, "Context fMHA requires >= 128 tokens per block"

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    if args.max_num_tokens is not None:
        assert args.enable_context_fmha or args.enable_context_fmha_fp32_acc

    if args.moe_num_experts and args.moe_top_k == 0:
        args.moe_top_k = 1
    # TODO: pass this moe_config to the model.
    args.moe_config = MoeConfig(
        args.moe_num_experts, args.moe_top_k, args.moe_tp_mode, args.moe_renorm_mode
    ).validate()

    logger.set_level(args.log_level)
    tik = time.time()
    _build_impl(rank, tensorrt_llm_model, args)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time building rank-{rank} engine: {t}")
