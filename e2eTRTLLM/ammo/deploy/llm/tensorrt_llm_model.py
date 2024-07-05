# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This module defines a tensorrt_llm based model for all LLMs we support inside AMMO.

Referrence impl in tensorrt_llm: tensorrt_llm/models/gpt/model.py.
"""
import inspect
from pathlib import Path
from typing import List

import numpy as np
import torch
from tensorrt_llm import Mapping, default_net, str_dtype_to_trt
from tensorrt_llm.functional import gather_last_token_logits
from tensorrt_llm.layers import AttentionParams, ColumnLinear, KeyValueCacheParams, LoraParams
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.module import Module, ModuleList

from ammo.torch.export import QUANTIZATION_FP8, ModelConfig

from .decoder import build_decoder_layer
from .quantization_utils import quantize_linear
from .tensorrt_llm_build import build
from .tensorrt_llm_utils import (
    build_embedding_from_config,
    build_layernorm_from_config,
    print_tensorrt_llm,
)


class ModelBuilder(Module):
    """A generic tensorrt_llm transformer model builder.

    We try to make this module builder as flexibile as possible to cover all transformer conversion usecases.
    """

    def __init__(self, model_config: ModelConfig):
        """Initializes the ModelBuilder from a model_config."""
        super().__init__()
        self.quantization = model_config.quantization
        self.rank = model_config.rank
        self.max_position_embeddings = model_config.max_position_embeddings
        self.hidden_act = model_config.hidden_act

        self._dtype_str = model_config.dtype
        self._dtype = str_dtype_to_trt(model_config.dtype)
        self._logits_dtype = str_dtype_to_trt("float16")
        self._kv_dtype = self._dtype
        # We only support tensor parallel so far.
        self._mapping = Mapping(
            world_size=model_config.tensor_parallel,
            rank=self.rank,
            gpus_per_node=model_config.tensor_parallel,
            tp_size=model_config.tensor_parallel,
            pp_size=1,
        )
        self._tp_size = self._mapping.tp_size
        self._tp_group = self._mapping.tp_group
        self._vocab_size = model_config.vocab_size
        self._hidden_size = model_config.hidden_size
        self._num_layers = len(model_config.layers)
        self._num_heads = model_config.num_attention_heads
        self._num_kv_heads = model_config.num_kv_heads

        # TODO: support use_prompt_tuning
        self.vocab_embedding = build_embedding_from_config(
            model_config.vocab_embedding,
            self._dtype,
            rank=self.rank,
            tensor_parallel=self._tp_size,
            instance_id=2 * self._num_layers,
        )
        self.positional_embedding = build_embedding_from_config(
            model_config.positional_embedding,
            self._dtype,
            rank=self.rank,
            tensor_parallel=self._tp_size,
        )

        self.ln_embed = build_layernorm_from_config(model_config.ln_embed, self._dtype)

        self.layers = ModuleList(
            [
                build_decoder_layer(
                    layer,
                    layer_id,
                    self._num_layers,
                    dtype=self._dtype,
                    quantization=model_config.quantization,
                    rank=self.rank,
                    tensor_parallel=self._tp_size,
                )
                for layer_id, layer in enumerate(model_config.layers)
            ]
        )

        # Backward compatibility
        if not model_config.ln_f:
            model_config.ln_f = model_config.final_layernorm

        self.ln_f = build_layernorm_from_config(model_config.ln_f, self._dtype)

        self.quant_mode = self.layers[0].quant_mode

        if self.quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt("int8")
        elif self.quant_mode.has_fp8_kv_cache():
            self._kv_dtype = str_dtype_to_trt("fp8")

    def forward(
        self,
        input_ids,
        position_ids,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        workspace=None,
        lora_params=None,
    ):
        """Forward function for the full model."""
        # TODO: support use_prompt_tuning
        x = self.vocab_embedding(input_ids, workspace=workspace)
        if hasattr(self, "positional_embedding") and self.positional_embedding:
            assert position_ids
            x = x + self.positional_embedding(position_ids, workspace=workspace)

        if hasattr(self, "ln_embed") and self.ln_embed:
            x = self.ln_embed(x)

        hidden_states = x

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        for layer_idx, (layer, past, pointer, host_pointer, max_attention_window_size) in enumerate(
            zip(
                self.layers,
                kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes,
            )
        ):
            lora_layer_params = None
            if lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            def _has_argument(method, argument_name):
                signature = inspect.signature(method)
                return argument_name in signature.parameters

            kwargs = {}
            if _has_argument(layer, "workspace"):
                kwargs["workspace"] = workspace
            elif _has_argument(layer, "all_reduce_workspace"):
                kwargs["all_reduce_workspace"] = workspace
            elif _has_argument(layer, "lora_layer_params"):
                kwargs["lora_layer_params"] = lora_layer_params
            elif _has_argument(layer.forward, "use_cache"):
                kwargs["use_cache"] = use_cache
            elif _has_argument(layer.forward, "attention_mask"):
                kwargs["attention_mask"] = attention_mask

            hidden_states = layer(
                hidden_states,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
                **kwargs,
            )
            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class LMHeadModelBuilder(ModelBuilder, GenerationMixin):
    """The implementation of the model builder with an LMHead."""

    def __init__(self, model_config: ModelConfig):
        """Initializes the LMHeadModelBuilder from a model_config."""
        super().__init__(model_config)

        share_weight = None
        if model_config.share_embedding_table:
            share_weight = self.vocab_embedding.weight

        self.lm_head = ColumnLinear(
            self._hidden_size,
            model_config.vocab_size_padded,
            # Bias is not avaialbe if lm_head uses shared weights from the embedding layer
            bias=share_weight is None and model_config.lm_head.bias is not None,
            dtype=self._dtype,
            tp_group=self._tp_group,
            tp_size=self._tp_size,
            gather_output=True,
            share_weight=share_weight,
        )
        if share_weight is not None:
            self.lm_head.weight = share_weight
        else:
            self.lm_head.weight.value = model_config.lm_head.weight
            if model_config.lm_head.bias is not None:
                self.lm_head.bias.value = model_config.lm_head.bias
        if model_config.quantization == QUANTIZATION_FP8 and share_weight is None:
            # We only quantize lm_head with FP8 for accuracy concern.
            self.lm_head = quantize_linear(
                self.lm_head, model_config.quantization, model_config.lm_head
            )

    def forward(
        self,
        input_ids,
        position_ids=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        workspace=None,
        lora_params=None,
    ):
        """Forward function for the full LMHead model."""
        hidden_states = super().forward(
            input_ids,
            position_ids,
            use_cache,
            attention_mask,
            kv_cache_params,
            attention_params,
            prompt_embedding_table,
            prompt_tasks,
            prompt_vocab_size,
            workspace,
            lora_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
        )

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output("logits", self._logits_dtype)

        if use_cache:
            if default_net().plugin_config.paged_kv_cache is False:
                for i, present in enumerate(presents):
                    present.mark_output(f"present_key_value_{i}", self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_new_tokens,
        use_cache,
        max_beam_width: int = 1,
        max_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
        gather_all_token_logits: bool = False,
        max_draft_len: int = 0,
        lora_target_modules: List[str] = None,
    ):
        """@brief: Prepare inputs Tensors for the model.

        The given sizes are used to determine the
        ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        """
        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net().plugin_config.use_custom_all_reduce
        use_lora_plugin = default_net().plugin_config.lora_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_new_tokens=max_new_tokens,
            num_kv_heads=self._num_kv_heads,
            head_size=head_size,
            num_layers=self._num_layers,
            kv_dtype=self._kv_dtype,
            num_heads=self._num_heads,
            dtype=self._dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            gather_all_token_logits=gather_all_token_logits,
            mapping=self._mapping,
            max_num_tokens=max_num_tokens,
            prompt_embedding_table_size=prompt_embedding_table_size,
            use_lora_plugin=use_lora_plugin,
            max_draft_len=max_draft_len,
            lora_target_modules=lora_target_modules,
        )

        return (
            model_inputs["input_ids"],
            model_inputs["position_ids"],
            True,
            model_inputs["last_token_ids"],
            model_inputs["attention_mask"],
            KeyValueCacheParams(
                past_key_value=model_inputs["past_key_value"],
                host_past_key_value_lengths=model_inputs["host_past_key_value_lengths"],
                host_max_attention_window_sizes=model_inputs["host_max_attention_window_sizes"],
                kv_cache_block_pointers=model_inputs["kv_cache_block_pointers_list"],
                host_kv_cache_block_pointers=model_inputs["host_kv_cache_block_pointers_list"],
                cache_indirection=model_inputs["cache_indirection"],
            ),
            AttentionParams(
                sequence_length=model_inputs["sequence_length"],
                context_lengths=model_inputs["context_lengths"],
                host_context_lengths=model_inputs["host_context_lengths"],
                max_context_length=max_input_len,
                host_request_types=model_inputs["host_request_types"],
            ),
            model_inputs["prompt_embedding_table"],
            model_inputs["tasks"],
            model_inputs["prompt_vocab_size"],
            model_inputs["all_reduce_workspace"],
            LoraParams(
                model_inputs["lora_ranks"],
                model_inputs["lora_weights_pointers"],
                host_context_lengths=model_inputs["host_context_lengths"],
                max_context_length=max_input_len,
                host_request_types=model_inputs["host_request_types"],
            ),
        )

    def build(
        self,
        output_dir: Path,
        timing_cache: str = "",
        log_level: str = "warning",
        max_batch_size: int = 1,
        max_input_len: int = 200,
        max_output_len: int = 200,
        max_beam_width: int = 1,
        inflight_batching: bool = False,
        enable_sparsity: bool = False,
        refit_engine_path: Path = "",
    ):
        """Builds the model and generate the tensorrt_llm engine.

        Args:
            timing_cache: the name of the tensorrt timing cache file inside the output_dir.
            log_level: the logging level.
            max_batch_size: the max batch size of the deployed model engine.
            max_input_len: the max length of the input tokens.
            max_output_len: the max length of the output tokens.
            max_beam_width: the max beam search width.
            output_dir: the output directory where we save the generated tensorrt_llm engine file.
            refit_engine_path: if provided, we try to refit the weights to the provided engine.
        """
        # Uncomment the following to print the network for debugging purpose.
        # self.print()

        # Number of GPUs has to match the tensor_parallel config if we are building in parallel.
        if self.rank >= torch.cuda.device_count():
            print(f"warning: Rank {self.rank} larger than GPUs available")
        if self._tp_size > torch.cuda.device_count():
            print(f"warning: Not enough GPUs locally, requesting {self._tp_size}")

        build(
            self,
            output_dir,
            self.rank,
            self._tp_size,
            self._dtype_str,
            timing_cache,
            log_level,
            max_batch_size,
            max_input_len,
            max_output_len,
            max_beam_width,
            torch.cuda.device_count(),
            quantization=self.quantization,
            inflight_batching=inflight_batching,
            enable_sparsity=enable_sparsity,
            refit_engine_path=refit_engine_path,
        )

    def print(self):
        """Debugging print of the tensorrt_llm network."""
        np.set_printoptions(threshold=36)
        print_tensorrt_llm(f"rank.{self.rank}", self)
