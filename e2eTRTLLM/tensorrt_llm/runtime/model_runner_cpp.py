# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from pathlib import Path
from typing import List, Optional, Union

import torch

from .. import profiler
from ..bindings import (DataType, GenerationInput, GenerationOutput,
                        GptJsonConfig, GptSession, GptSessionConfig,
                        KvCacheConfig, PromptTuningParams)
from ..bindings import SamplingConfig as GptSamplingConfig
from ..bindings import WorldConfig
from ..logger import logger
from ..mapping import Mapping
from .generation import (LogitsProcessor, LoraManager, SamplingConfig,
                         StoppingCriteria)
from .model_runner import ModelRunnerMixin

_bindings_dtype_to_torch_dtype_dict = {
    DataType.FLOAT: torch.float,
    DataType.HALF: torch.half,
    DataType.INT8: torch.int8,
    DataType.INT32: torch.int32,
    DataType.BOOL: torch.bool,
    DataType.UINT8: torch.uint8,
    DataType.BF16: torch.bfloat16,
    DataType.INT64: torch.int64
}


class ModelRunnerCpp(ModelRunnerMixin):
    """
    An interface class that wraps GptSession and provides generation methods.
    """

    def __init__(self,
                 session: GptSession,
                 max_batch_size: int,
                 max_input_len: int,
                 max_seq_len: int,
                 max_beam_width: int,
                 lora_manager: Optional[LoraManager] = None) -> None:
        """
        Create a ModelRunnerCpp instance.
        You are recommended to use the from_dir method to load the engine and create a ModelRunnerCpp instance.

        Args:
            session (GenerationSession):
                The TensorRT session created from an engine.
            max_batch_size (int):
                The maximum batch size allowed for the input.
            max_input_len (int):
                The maximum input length allowed for the input.
            max_seq_len (int):
                The maximum sequence length (input + generated tokens).
            max_beam_width (int):
                The maximum beam width.
            lora_manager (LoraManager):
                The LoRA manager to handle LoRA weights.
        """
        self.session = session
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width
        self.lora_manager = lora_manager
        self.mapping = Mapping(
            world_size=session.world_config.tensor_parallelism *
            session.world_config.pipeline_parallelism,
            rank=session.world_config.rank,
            gpus_per_node=session.world_config.gpus_per_node,
            tp_size=session.world_config.tensor_parallelism,
            pp_size=session.world_config.pipeline_parallelism)
        self.session_config = GptSessionConfig(max_batch_size=max_batch_size,
                                               max_beam_width=max_beam_width,
                                               max_sequence_length=max_seq_len)

    @classmethod
    def from_dir(cls,
                 engine_dir: str,
                 lora_dir: Optional[str] = None,
                 rank: int = 0,
                 max_batch_size: Optional[int] = None,
                 max_input_len: Optional[int] = None,
                 max_output_len: Optional[int] = None,
                 max_beam_width: Optional[int] = None,
                 max_attention_window_size: Optional[int] = None,
                 sink_token_length: Optional[int] = None,
                 free_gpu_memory_fraction: Optional[float] = None,
                 debug_mode: bool = False,
                 lora_ckpt_source: str = "hf") -> 'ModelRunnerCpp':
        """
        Create a ModelRunnerCpp instance from an engine directory.

        Args:
            engine_dir (str):
                The directory that contains the serialized engine files and config files.
            lora_dir (str):
                The directory that contains LoRA weights.
            rank (int):
                The runtime rank id.
            max_batch_size (int):
                The runtime batch size limit. If max_batch_size is not None, it should not
                be larger than the engine's max_batch_size; otherwise, the engine's max_batch_size
                will be used.
            max_input_len (int):
                The runtime input length limit. If max_input_len is not None, it should not
                be larger than the engine's max_input_len; otherwise, the engine's max_input_len
                will be used.
            max_output_len (int):
                The runtime output length limit. If max_output_len is not None, it should not
                be larger than the engine's max_output_len; otherwise, the engine's max_output_len
                will be used.
            max_beam_width (int):
                The runtime beam width limit. If max_beam_width is not None, it should not
                be larger than the engine's max_beam_width; otherwise, the engine's max_beam_width
                will be used.
            max_attention_window_size (int):
                The attention window size that controls the sliding window attention / cyclic kv cache behavior.
            sink_token_length (int) :
                The sink token length, default=0.
            free_gpu_memory_fraction (float) :
                Free GPU memory fraction that KV cache used.
            debug_mode (bool):
                Whether or not to turn on the debug mode.
            lora_ckpt_source (str):
                Source of checkpoint. Should be one of ['hf', 'nemo'].
        Returns:
            ModelRunnerCpp: An instance of ModelRunnerCpp.
        """
        # session setup
        config_path = Path(engine_dir) / "config.json"
        json_config = GptJsonConfig.parse_file(config_path)
        model_config = json_config.model_config

        tp_size = json_config.tensor_parallelism
        pp_size = json_config.pipeline_parallelism
        world_config = WorldConfig.mpi(tensor_parallelism=tp_size,
                                       pipeline_parallelism=pp_size)
        assert rank == world_config.rank
        engine_filename = json_config.engine_filename(world_config)
        serialize_path = Path(engine_dir) / engine_filename

        profiler.start('load tensorrt_llm engine')
        if max_beam_width is None:
            max_beam_width = model_config.max_beam_width
        else:
            assert max_beam_width <= model_config.max_beam_width
        if max_batch_size is None:
            max_batch_size = model_config.max_batch_size
        else:
            assert max_batch_size <= model_config.max_batch_size
        if max_input_len is None:
            max_input_len = model_config.max_input_len
        else:
            assert max_input_len <= model_config.max_input_len
        if max_output_len is None:
            max_seq_len = model_config.max_seq_len
        else:
            max_seq_len = max_input_len + max_output_len
            assert max_seq_len <= model_config.max_seq_len
        session_config = GptSessionConfig(max_batch_size=max_batch_size,
                                          max_beam_width=max_beam_width,
                                          max_sequence_length=max_seq_len)
        session_config.kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            max_attention_window=max_attention_window_size,
            sink_token_length=sink_token_length)
        session = GptSession(config=session_config,
                             model_config=model_config,
                             world_config=world_config,
                             engine_file=str(serialize_path))
        profiler.stop('load tensorrt_llm engine')
        loading_time = profiler.elapsed_time_in_sec("load tensorrt_llm engine")
        logger.info(f'Load engine takes: {loading_time} sec')

        # TODO: LoRA not supported
        if lora_dir is not None:
            raise RuntimeError("LoRA is not supported in C++ session.")
        return cls(session,
                   lora_manager=None,
                   max_batch_size=max_batch_size,
                   max_input_len=max_input_len,
                   max_seq_len=max_seq_len,
                   max_beam_width=max_beam_width)

    @property
    def dtype(self) -> torch.dtype:
        bindings_dtype = self.session.model_config.data_type
        return _bindings_dtype_to_torch_dtype_dict[bindings_dtype]

    @property
    def vocab_size(self) -> int:
        return self.session.model_config.vocab_size

    @property
    def vocab_size_padded(self) -> int:
        return self.session.model_config.vocab_size_padded(
            self.session.world_config.size)

    @property
    def hidden_size(self) -> int:
        return self.session.model_config.hidden_size

    @property
    def num_heads(self) -> int:
        return self.session.model_config.num_heads

    @property
    def num_layers(self) -> int:
        return self.session.model_config.num_layers(
            self.session.world_config.pipeline_parallelism)

    @property
    def max_sequence_length(self) -> int:
        return self.max_seq_len

    @property
    def remove_input_padding(self) -> bool:
        return self.session.model_config.use_packed_input

    @property
    def max_prompt_embedding_table_size(self) -> int:
        return self.session.model_config.max_prompt_embedding_table_size

    @property
    def gather_context_logits(self) -> bool:
        return self.session.model_config.compute_context_logits

    @property
    def gather_generation_logits(self) -> bool:
        return self.session.model_config.compute_generation_logits

    def generate(self,
                 batch_input_ids: List[torch.Tensor],
                 sampling_config: Optional[SamplingConfig] = None,
                 prompt_table: Optional[Union[str, torch.Tensor]] = None,
                 prompt_tasks: Optional[str] = None,
                 lora_uids: Optional[list] = None,
                 streaming: bool = False,
                 stopping_criteria: Optional[StoppingCriteria] = None,
                 logits_processor: Optional[LogitsProcessor] = None,
                 **kwargs) -> Union[torch.Tensor, dict]:
        """
        Generates sequences of token ids.
        The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
        You can override any sampling_config's attributes by passing corresponding parameters.

        Args:
            batch_input_ids (List[torch.Tensor]):
                A list of input id tensors. Each tensor is of shape (sequence_length, ).
            sampling_config (SamplingConfig):
                The sampling configuration to be used as base parametrization for the generation call.
                The passed **kwargs matching the sampling_config's attributes will override them.
                If the sampling_config is not provided, a default will be used.
            prompt_table (str or torch.Tensor):
                The file path of prompt table (.npy format, exported by nemo_prompt_convert.py) or the prompt table itself.
            prompt_tasks (str):
                The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
            lora_uids (list):
                The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
            streaming (bool):
                Whether or not to use streaming mode for generation.
            stopping_criteria (StoppingCriteria):
                Custom stopping criteria.
            logits_processor (LogitsProcessor):
                Custom logits processors.
            kwargs (Dict[str, Any]:
                Ad hoc parametrization of sampling_config.
                The passed **kwargs matching the sampling_config's attributes will override them.
        Returns:
            torch.Tensor or dict:
                If return_dict=False, the method returns generated output_ids.
                If return_dict=True, the method returns a dict of output_ids,
                sequence_lengths (if sampling_config.output_sequence_lengths=True),
                context_logits and generation_logits (if self.gather_context_logits=True and
                self.gather_generation_logits=True, respectively).
        """
        if sampling_config is None:
            sampling_config = SamplingConfig(end_id=None, pad_id=None)
        else:
            sampling_config = copy.deepcopy(sampling_config)
        sampling_config.update(**kwargs)
        self._check_inputs(batch_input_ids, sampling_config)
        batch_size = len(batch_input_ids)
        gpt_sampling_config = _populate_sampling_config(sampling_config,
                                                        batch_size)
        if lora_uids is not None:
            raise RuntimeError("LoRA is not supported in C++ session.")
        if streaming:
            raise RuntimeError("Streaming is not supported in C++ session.")
        if stopping_criteria is not None:
            raise RuntimeError(
                "Stopping criteria is not supported in C++ session.")
        if logits_processor is not None:
            raise RuntimeError(
                "Logits processor is not supported in C++ session.")

        batch_input_ids, input_lengths = self._prepare_inputs(
            batch_input_ids, sampling_config.pad_id)

        batch_input_ids = batch_input_ids.cuda()
        input_lengths = input_lengths.cuda()
        generation_input = GenerationInput(sampling_config.end_id,
                                           sampling_config.pad_id,
                                           batch_input_ids, input_lengths,
                                           self.remove_input_padding)
        generation_input.max_new_tokens = sampling_config.max_new_tokens
        generation_input.bad_words_list = sampling_config.bad_words_list
        generation_input.stop_words_list = sampling_config.stop_words_list

        if self.max_prompt_embedding_table_size > 0:
            ptuning_kwargs = self._prepare_ptuning(prompt_table, prompt_tasks,
                                                   batch_size)
            generation_input.prompt_tuning_params = PromptTuningParams(
                **ptuning_kwargs)
            generation_input.prompt_tuning_params.prompt_tuning_enabled = [
                True
            ] * batch_size

        cuda_device = torch.device(self.session.device)
        output_ids = torch.empty(
            (batch_size, sampling_config.num_beams, self.max_sequence_length),
            dtype=torch.int32,
            device=cuda_device)
        output_lengths = torch.empty((batch_size, sampling_config.num_beams),
                                     dtype=torch.int32,
                                     device=cuda_device)
        generation_output = GenerationOutput(output_ids, output_lengths)
        if sampling_config.output_cum_log_probs:
            generation_output.cum_log_probs = torch.empty(
                (batch_size, sampling_config.num_beams),
                dtype=torch.float32,
                device=cuda_device)
        if sampling_config.output_log_probs:
            generation_output.log_probs = torch.empty(
                (batch_size, sampling_config.num_beams,
                 self.max_input_len + sampling_config.max_new_tokens),
                dtype=torch.float32,
                device=cuda_device)
        if self.gather_context_logits:
            generation_output.context_logits = torch.empty(
                (batch_size, self.max_input_len, self.vocab_size_padded),
                device=cuda_device)
        if self.gather_generation_logits:
            generation_output.generation_logits = torch.zeros(
                (batch_size, sampling_config.num_beams,
                 sampling_config.max_new_tokens, self.vocab_size_padded),
                device=cuda_device)

        self.session.generate(generation_output, generation_input,
                              gpt_sampling_config)
        if sampling_config.return_dict:
            outputs = {'output_ids': generation_output.ids}
            if sampling_config.output_sequence_lengths:
                outputs['sequence_lengths'] = generation_output.lengths
            if sampling_config.output_cum_log_probs:
                outputs['cum_log_probs'] = generation_output.cum_log_probs
            if sampling_config.output_log_probs:
                outputs['log_probs'] = generation_output.log_probs
            if self.gather_context_logits:
                outputs['context_logits'] = generation_output.context_logits
            if self.gather_generation_logits:
                outputs[
                    'generation_logits'] = generation_output.generation_logits
            outputs = self._prepare_outputs(outputs, input_lengths)
        else:
            outputs = generation_output.ids
        return outputs


def _populate_sampling_config(sampling_config: SamplingConfig,
                              batch_size: int) -> GptSamplingConfig:
    gpt_sampling_config = GptSamplingConfig(sampling_config.num_beams)

    if isinstance(sampling_config.beam_search_diversity_rate, torch.Tensor):
        assert sampling_config.beam_search_diversity_rate.dtype == torch.float32, f"sampling_config.beam_search_diversity_rate.dtype ({sampling_config.beam_search_diversity_rate.dtype}) must be torch.float32"
        assert sampling_config.beam_search_diversity_rate.shape[
            0] == batch_size, f"sampling_config.beam_search_diversity_rate.shape[0] ({sampling_config.beam_search_diversity_rate.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.beam_search_diversity_rate = sampling_config.beam_search_diversity_rate.tolist(
        )
    elif sampling_config.beam_search_diversity_rate is not None:
        gpt_sampling_config.beam_search_diversity_rate = [
            sampling_config.beam_search_diversity_rate
        ]
    else:
        gpt_sampling_config.beam_search_diversity_rate = None

    if isinstance(sampling_config.length_penalty, torch.Tensor):
        assert sampling_config.length_penalty.dtype == torch.float32, f"sampling_config.length_penalty.dtype ({sampling_config.length_penalty.dtype}) must be torch.float32"
        assert sampling_config.length_penalty.shape[
            0] == batch_size, f"sampling_config.length_penalty.shape[0] ({sampling_config.length_penalty.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.length_penalty = sampling_config.length_penalty.tolist(
        )
    else:
        gpt_sampling_config.length_penalty = [sampling_config.length_penalty]

    if isinstance(sampling_config.early_stopping, torch.Tensor):
        assert sampling_config.early_stopping.dtype == torch.int32, f"sampling_config.early_stopping.dtype ({sampling_config.early_stopping.dtype}) must be torch.int32"
        assert sampling_config.early_stopping.shape[
            0] == batch_size, f"sampling_config.early_stopping.shape[0] ({sampling_config.early_stopping.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.early_stopping = sampling_config.early_stopping.tolist(
        )
    else:
        gpt_sampling_config.early_stopping = [sampling_config.early_stopping]

    if isinstance(sampling_config.min_length, torch.Tensor):
        assert sampling_config.min_length.dtype == torch.int32, f"sampling_config.min_length.dtype ({sampling_config.min_length.dtype}) must be torch.int32"
        assert sampling_config.min_length.shape[
            0] == batch_size, f"sampling_config.min_length.shape[0] ({sampling_config.min_length.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.min_length = sampling_config.min_length.tolist()
    else:
        gpt_sampling_config.min_length = [sampling_config.min_length]

    if isinstance(sampling_config.presence_penalty, torch.Tensor):
        assert sampling_config.presence_penalty.dtype == torch.float32, f"sampling_config.presence_penalty.dtype ({sampling_config.presence_penalty.dtype}) must be torch.float32"
        assert sampling_config.presence_penalty.shape[
            0] == batch_size, f"sampling_config.presence_penalty.shape[0] ({sampling_config.presence_penalty.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.presence_penalty = sampling_config.presence_penalty.tolist(
        )
    elif sampling_config.presence_penalty == 0.0:
        gpt_sampling_config.presence_penalty = None
    else:
        gpt_sampling_config.presence_penalty = [
            sampling_config.presence_penalty
        ]

    if isinstance(sampling_config.frequency_penalty, torch.Tensor):
        assert sampling_config.frequency_penalty.dtype == torch.float32, f"sampling_config.frequency_penalty.dtype ({sampling_config.frequency_penalty.dtype}) must be torch.float32"
        assert sampling_config.frequency_penalty.shape[
            0] == batch_size, f"sampling_config.frequency_penalty.shape[0] ({sampling_config.frequency_penalty.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.frequency_penalty = sampling_config.frequency_penalty.tolist(
        )
    elif sampling_config.frequency_penalty == 0.0:
        gpt_sampling_config.frequency_penalty = None
    else:
        gpt_sampling_config.frequency_penalty = [
            sampling_config.frequency_penalty
        ]

    if isinstance(sampling_config.random_seed, torch.Tensor):
        assert sampling_config.random_seed.dtype == torch.int64, f"sampling_config.random_seed.dtype ({sampling_config.random_seed.dtype}) must be torch.int64"
        assert sampling_config.random_seed.shape[
            0] == batch_size, f"sampling_config.random_seed.shape[0] ({sampling_config.random_seed.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.random_seed = sampling_config.random_seed.tolist()
    elif sampling_config.random_seed is not None:
        gpt_sampling_config.random_seed = [sampling_config.random_seed]
    else:
        gpt_sampling_config.random_seed = None

    if isinstance(sampling_config.repetition_penalty, torch.Tensor):
        assert sampling_config.repetition_penalty.dtype == torch.float32, f"sampling_config.repetition_penalty.dtype ({sampling_config.repetition_penalty.dtype}) must be torch.float32"
        assert sampling_config.repetition_penalty.shape[
            0] == batch_size, f"sampling_config.repetition_penalty.shape[0] ({sampling_config.repetition_penalty.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.repetition_penalty = sampling_config.repetition_penalty.tolist(
        )
    elif sampling_config.repetition_penalty == 1.0:
        gpt_sampling_config.repetition_penalty = None
    else:
        gpt_sampling_config.repetition_penalty = [
            sampling_config.repetition_penalty
        ]

    if isinstance(sampling_config.temperature, torch.Tensor):
        assert sampling_config.temperature.dtype == torch.float32, f"sampling_config.temperature.dtype ({sampling_config.temperature.dtype}) must be torch.float32"
        assert sampling_config.temperature.shape[
            0] == batch_size, f"sampling_config.temperature.shape[0] ({sampling_config.temperature.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.temperature = sampling_config.temperature.tolist()
    else:
        gpt_sampling_config.temperature = [sampling_config.temperature]

    if isinstance(sampling_config.top_k, torch.Tensor):
        assert sampling_config.top_k.dtype == torch.int32, f"sampling_config.top_k.dtype ({sampling_config.top_k.dtype}) must be torch.int32"
        assert sampling_config.top_k.shape[
            0] == batch_size, f"sampling_config.top_k.shape[0] ({sampling_config.top_k.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.top_k = sampling_config.top_k.tolist()
    else:
        gpt_sampling_config.top_k = [sampling_config.top_k]

    if isinstance(sampling_config.top_p, torch.Tensor):
        assert sampling_config.top_p.dtype == torch.float32, f"sampling_config.top_p.dtype ({sampling_config.top_p.dtype}) must be torch.float32"
        assert sampling_config.top_p.shape[
            0] == batch_size, f"sampling_config.top_p.shape[0] ({sampling_config.top_p.shape[0]}) must equal to batch_size ({batch_size})"
        gpt_sampling_config.top_p = sampling_config.top_p.tolist()
    else:
        gpt_sampling_config.top_p = [sampling_config.top_p]

    if sampling_config.top_p_decay is not None:
        gpt_sampling_config.top_p_decay = sampling_config.top_p_decay.tolist()
    if sampling_config.top_p_min is not None:
        gpt_sampling_config.top_p_min = sampling_config.top_p_min.tolist()
    if sampling_config.top_p_reset_ids is not None:
        gpt_sampling_config.top_p_reset_ids = sampling_config.top_p_reset_ids.tolist(
        )
    return gpt_sampling_config
