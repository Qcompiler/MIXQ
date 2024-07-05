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
from collections import OrderedDict
from typing import List, Optional

import tensorrt as trt

from ..._common import default_net
from ..._utils import str_dtype_to_trt
from ...functional import (Tensor, arange, cast, concat, expand,
                           gather_last_token_logits, shape, unsqueeze)
from ...layers import (Embedding, LayerNorm, Linear, Mamba, MambaParameters,
                       RmsNorm)
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin
from ..modeling_utils import PretrainedConfig, PretrainedModel


class MambaLayer(Module):

    def __init__(self, config: PretrainedConfig, last_layer=False):
        super().__init__()
        self.dtype = config.dtype
        self.residual_in_fp32 = config.residual_in_fp32
        self.last_layer = last_layer

        self.ssm = Mamba(config.hidden_size,
                         **config.ssm_cfg,
                         dtype=config.dtype)
        if config.rms_norm:
            self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                           eps=config.norm_epsilon,
                                           dtype=config.dtype)
        else:
            self.input_layernorm = LayerNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                residual: Tensor,
                conv_state: Tensor,
                ssm_state: Tensor,
                host_request_types: Tensor,
                host_context_lengths: Tensor,
                last_token_ids: Tensor,
                slot_mapping: Optional[Tensor] = None,
                conv_indices: Optional[Tensor] = None):

        hidden_states = self.input_layernorm(hidden_states)

        ssm_out, present_conv, present_ssm = self.ssm(
            hidden_states,
            conv_state=conv_state,
            ssm_state=ssm_state,
            host_request_types=host_request_types,
            last_token_ids=last_token_ids,
            host_context_lengths=host_context_lengths,
            slot_mapping=slot_mapping,
            conv_indices=conv_indices)
        if self.residual_in_fp32:
            residual = residual + cast(ssm_out, 'float32')
            hidden_states = cast(residual, self.dtype)
        else:
            residual = residual + ssm_out
            hidden_states = residual

        if self.last_layer:
            return hidden_states, None, present_conv, present_ssm
        else:
            return hidden_states, residual, present_conv, present_ssm


class MambaModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.d_conv = config.ssm_cfg['d_conv']
        self.d_inner = int(config.ssm_cfg['expand'] * config.hidden_size)
        n_layer = config.num_hidden_layers
        self.residual_in_fp32 = config.residual_in_fp32
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple)
        self.vocab_embedding = Embedding(config.vocab_size,
                                         config.hidden_size,
                                         dtype=config.dtype)
        self.layers = ModuleList([
            MambaLayer(config, last_layer=i == n_layer - 1)
            for i in range(n_layer)
        ])
        if config.rms_norm:
            self.norm_f = RmsNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  dtype=config.dtype)
        else:
            self.norm_f = LayerNorm(normalized_shape=config.hidden_size,
                                    eps=config.norm_epsilon,
                                    dtype=config.dtype)

    def forward(self,
                input_ids,
                conv_states,
                ssm_states,
                host_request_types,
                host_context_lengths,
                last_token_ids,
                slot_mapping: Optional[Tensor] = None):
        hidden_states = self.vocab_embedding(input_ids)

        # Get conv state indices
        indices = None
        if not default_net().plugin_config.mamba_conv1d_plugin:
            batch_size = shape(input_ids, 0)
            indices = expand(
                unsqueeze(arange(0, self.d_conv - 1, dtype='int32'), 0),
                concat([batch_size, self.d_conv - 1]))
            offsets = expand(unsqueeze(last_token_ids, 1),
                             concat([batch_size, self.d_conv - 1]))
            indices = unsqueeze(indices + offsets, 1)
            indices = expand(
                indices, concat([batch_size, self.d_inner, self.d_conv - 1]))

        residual = cast(hidden_states,
                        'float32') if self.residual_in_fp32 else hidden_states
        hidden_values = [hidden_states, residual]
        present_convs, present_ssms = [], []
        for layer, past_conv, past_ssm in zip(self.layers, conv_states,
                                              ssm_states):
            hidden_values = layer(hidden_values[0], hidden_values[1], past_conv,
                                  past_ssm, host_request_types,
                                  host_context_lengths, last_token_ids,
                                  slot_mapping, indices)
            present_convs.append(hidden_values[2])
            present_ssms.append(hidden_values[3])
        hidden_states = hidden_values[0]
        hidden_states = self.norm_f(hidden_states)
        return hidden_states, tuple(present_convs), tuple(present_ssms)


class MambaLMHeadModel(PretrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        dtype = config.dtype
        logits_dtype = config.logits_dtype
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype

        self.ssm_cfg = MambaParameters(**config.ssm_cfg)
        self.d_inner = self.ssm_cfg.expand * config.hidden_size
        self.d_conv = self.ssm_cfg.d_conv
        self.d_state = self.ssm_cfg.d_state
        self.config = config
        self.gather_context_logits = False

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self.backbone = MambaModel(config)
        self.lm_head = Linear(config.hidden_size,
                              config.vocab_size,
                              bias=False,
                              dtype=dtype,
                              gather_output=False)

    def __post_init__(self):
        return

    def forward(self,
                input_ids,
                conv_states,
                ssm_states,
                host_request_types,
                host_context_lengths,
                last_token_ids,
                slot_mapping: Optional[Tensor] = None):
        hidden_states, present_convs, present_ssms = self.backbone(
            input_ids, conv_states, ssm_states, host_request_types,
            host_context_lengths, last_token_ids, slot_mapping)

        if not self.gather_context_logits:
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._logits_dtype)
        if not default_net().plugin_config.paged_state:
            for i, present_conv in enumerate(present_convs):
                present_conv.mark_output(f'present_conv_state_{i}', self.dtype)
            for i, present_ssm in enumerate(present_ssms):
                present_ssm.mark_output(f'present_ssm_state_{i}', self.dtype)

        return (lm_logits, present_convs, present_ssms)

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_seq_len,
                       use_cache,
                       max_beam_width: int = 1,
                       max_num_tokens: int = None,
                       opt_num_tokens: int = None,
                       prompt_embedding_table_size: int = 0,
                       max_draft_len: int = 0,
                       gather_context_logits: bool = False,
                       gather_generation_logits: bool = False,
                       lora_target_modules: List[str] = None):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_mamba_conv1d_plugin = default_net(
        ).plugin_config.mamba_conv1d_plugin
        batch_range = [GenerationMixin.default_range(max_batch_size)]
        self.gather_context_logits = gather_context_logits
        if remove_input_padding:
            assert use_mamba_conv1d_plugin, "mamba_conv1d_plugin is needed to support remove_input_padding"
            max_num_tokens = max(
                max_input_len * max_batch_size,
                max_beam_width * (max_draft_len + 1) * max_batch_size)
            if opt_num_tokens is None:
                opt_num_tokens = max_beam_width * (max_draft_len +
                                                   1) * max_batch_size
            num_tokens_range = [[1, opt_num_tokens, max_num_tokens]]
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([
                                   ('num_tokens', num_tokens_range),
                               ]))
        else:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', batch_range),
                                   ('input_len', [[1, 1, max_input_len]]),
                               ]))
        conv_states = []
        ssm_states = []
        if use_mamba_conv1d_plugin:
            conv_state_dim_range = OrderedDict([
                ('batch_size', batch_range),
                ('kernel_size', [self.d_conv - 1]),
                ('dim_size', [self.d_inner]),
            ])
        else:
            conv_state_dim_range = OrderedDict([
                ('batch_size', batch_range),
                ('dim_size', [self.d_inner]),
                ('kernel_size', [self.d_conv - 1]),
            ])

        ssm_state_dim_range = OrderedDict([
            ('batch_size', batch_range),
            ('state_size', [self.d_state]),
            ('dim_size', [self.d_inner]),
        ])
        one_dim_range = OrderedDict([
            ('buffer_count', [1]),
        ])

        for i in range(self.config.num_hidden_layers):
            if default_net().plugin_config.paged_state:
                conv_state = Tensor(name=f'conv_state_ptr_{i}',
                                    dtype=str_dtype_to_trt('int64'),
                                    shape=[1],
                                    dim_range=one_dim_range)

                ssm_state = Tensor(name=f'ssm_state_ptr_{i}',
                                   dtype=str_dtype_to_trt('int64'),
                                   shape=[1],
                                   dim_range=one_dim_range)
            else:
                if use_mamba_conv1d_plugin:
                    conv_state = Tensor(
                        name=f'past_conv_state_{i}',
                        dtype=self.dtype,
                        shape=[-1, self.d_conv - 1, self.d_inner],
                        dim_range=conv_state_dim_range)
                else:
                    conv_state = Tensor(
                        name=f'past_conv_state_{i}',
                        dtype=self.dtype,
                        shape=[-1, self.d_inner, self.d_conv - 1],
                        dim_range=conv_state_dim_range)

                ssm_state = Tensor(name=f'past_ssm_state_{i}',
                                   dtype=self.dtype,
                                   shape=[-1, self.d_state, self.d_inner],
                                   dim_range=ssm_state_dim_range)

            conv_states.append(conv_state)
            ssm_states.append(ssm_state)

        host_request_types = Tensor(
            name='host_request_types',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_size', batch_range)]),
        )

        host_context_lengths = Tensor(
            name='host_context_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_size', batch_range)]),
        )

        last_token_ids = None
        if not gather_context_logits:
            last_token_ids = Tensor(
                name='last_token_ids',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([
                    ('batch_size', batch_range),
                ]),
            )

        return_dict = {
            'input_ids': input_ids,
            'conv_states': conv_states,
            'ssm_states': ssm_states,
            'host_request_types': host_request_types,
            'last_token_ids': last_token_ids,
            'host_context_lengths': host_context_lengths,
        }

        if default_net().plugin_config.paged_state:
            slot_mapping = Tensor(
                name='slot_mapping',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', batch_range)]),
            )
            return_dict['slot_mapping'] = slot_mapping

        return return_dict
