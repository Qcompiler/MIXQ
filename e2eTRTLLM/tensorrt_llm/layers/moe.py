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
from dataclasses import dataclass
from enum import IntEnum
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._utils import str_dtype_to_trt

from .._common import default_net, default_trtnet
from ..functional import (_create_tensor, allreduce, cast, is_gated_activation,
                          non_gated_version, softmax, sum, topk)
from ..layers import MLP, GatedMLP
from ..module import Module
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from ..quantization import QuantMode
from .linear import RowLinear

activation_str_to_int_map = {
    # [WARNING] Keep the below in sync with cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h
    "gelu": 0,
    "gelu_new": 0,
    "relu": 1,
    "silu": 2,
    "swiglu": 3,
    "geglu": 4,
    "identity": 5,
}


@dataclass
class MoeConfig:
    # [WARNING] Keep the below in sync with cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h
    class ParallelismMode(IntEnum):
        NONE = 0
        EXPERT_PARALLEL = 1
        TENSOR_PARALLEL = 2

    class ExpertScaleNormalizationMode(IntEnum):
        NONE = 0
        RENORMALIZE = 1

    num_experts: int = 0
    top_k: int = 0
    tp_mode: ParallelismMode = ParallelismMode.TENSOR_PARALLEL
    normalization_mode: ExpertScaleNormalizationMode = ExpertScaleNormalizationMode.RENORMALIZE

    def validate(self) -> "MoeConfig":
        if (self.num_experts == 0) != (self.top_k == 0):
            raise ValueError(
                "Both or neither MoeConfig's num_experts and top_k must be set to 0"
            )
        return self

    def has_moe(self) -> bool:
        return self.num_experts > 1


def _moe_plugin(moe_config,
                hidden_states,
                routing,
                finished,
                expert_weight_1,
                expert_weight_2,
                expert_bias_1,
                expert_bias_2,
                expert_scale_1,
                expert_scale_2,
                hidden_size,
                ffn_hidden_size,
                act_fn,
                dtype,
                weight_dtype,
                quant_mode=QuantMode(0),
                tp_size=1,
                tp_rank=0):
    if isinstance(dtype, str):
        dtype = str_dtype_to_trt(dtype)

    if isinstance(weight_dtype, str):
        weight_dtype = str_dtype_to_trt(weight_dtype)
    # Create the plugin with our required state
    num_experts = moe_config.num_experts
    # We pass the full number of experts (not divided by tp_size) even for EP mode
    p_num_experts = trt.PluginField("number_of_experts",
                                    np.array(num_experts, dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    p_top_k = trt.PluginField("top_k", np.array(moe_config.top_k,
                                                dtype=np.int32),
                              trt.PluginFieldType.INT32)
    p_expert_hidden_size = trt.PluginField(
        "expert_hidden_size", np.array(hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_expert_inter_size = trt.PluginField(
        "expert_inter_size", np.array(ffn_hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_activation_type = trt.PluginField(
        "activation_type",
        np.array(activation_str_to_int_map[act_fn], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_type_id = trt.PluginField("type_id", np.array([int(dtype)],
                                                    dtype=np.int32),
                                trt.PluginFieldType.INT32)

    p_weight_type_id = trt.PluginField(
        "weight_type_id", np.array([int(weight_dtype)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_quant_mode = trt.PluginField("quant_mode",
                                   np.array([int(quant_mode)], dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    p_use_finished = trt.PluginField(
        "use_finished", np.array([int(finished is not None)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_use_bias = trt.PluginField(
        "use_bias", np.array([int(expert_bias_1 is not None)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_tp_size = trt.PluginField("tp_size", np.array(tp_size, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_tp_rank = trt.PluginField("tp_rank", np.array(tp_rank, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_parallelism_mode = trt.PluginField(
        "parallelism_mode", np.array(moe_config.tp_mode, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_normalization_mode = trt.PluginField(
        "normalization_mode",
        np.array(moe_config.normalization_mode, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        p_num_experts, p_top_k, p_expert_hidden_size, p_expert_inter_size,
        p_activation_type, p_type_id, p_weight_type_id, p_quant_mode,
        p_use_finished, p_use_bias, p_tp_size, p_tp_rank, p_parallelism_mode,
        p_normalization_mode
    ])

    # Create the plugin with our constant inputs to the constructor
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixtureOfExperts', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    moe_plugin = plugin_creator.create_plugin("mixture_of_experts", pfc)

    # Instantiate the plugin with our specific inputs
    plugin_inputs = [
        hidden_states, routing, expert_weight_1.value, expert_weight_2.value
    ]

    if expert_bias_1:
        assert expert_bias_2
        plugin_inputs += [expert_bias_1.value, expert_bias_2.value]

    if finished is not None:
        plugin_inputs += [finished]

    # Add conditional inputs
    if expert_scale_1 is not None:
        assert expert_scale_2
        plugin_inputs += [expert_scale_1.value, expert_scale_2.value]

    plugin_inputs = [i.trt_tensor for i in plugin_inputs]
    layer = default_trtnet().add_plugin_v2(plugin_inputs, moe_plugin)
    if not default_net().strongly_typed:
        for ii in range(layer.num_inputs):
            if layer.get_input(ii).dtype == str_dtype_to_trt("int8"):
                layer.get_input(ii).set_dynamic_range(-127, 127)
    output = _create_tensor(layer.get_output(0), layer)
    return output


class MixtureOfExperts(Module):

    def __init__(self,
                 moe_config: MoeConfig,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 hidden_act: str,
                 bias: bool = True,
                 dtype=None,
                 tp_group: List[int] = None,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 quant_mode=QuantMode(0)):
        super().__init__()

        self.moe_config = moe_config
        self.num_experts = moe_config.num_experts
        self.top_k = moe_config.top_k

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.dtype = dtype
        self.weight_dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.quant_mode = quant_mode
        self.has_bias = bias

        self.experts_per_node = self.num_experts
        self.tp_mode = moe_config.tp_mode
        if moe_config.tp_mode == MoeConfig.ParallelismMode.EXPERT_PARALLEL:
            if self.num_experts % self.tp_size != 0:
                raise ValueError(
                    f"MixtureOfExperts - Number of experts {self.num_experts} is not a multiple of EP size {self.tp_size}"
                )
            self.experts_per_node = self.experts_per_node // tp_size

        elif moe_config.tp_mode == MoeConfig.ParallelismMode.TENSOR_PARALLEL:
            if self.ffn_hidden_size % self.tp_size != 0:
                raise ValueError(
                    f"MixtureOfExperts - FFN Hidden Size {self.ffn_hidden_size} is not a multiple of TP size {self.tp_size}"
                )
            self.ffn_hidden_size = self.ffn_hidden_size // tp_size

        if quant_mode.is_weight_only():
            self.weight_dtype = trt.int8

        # Since output dimension is usually low (in the order of 10s), no TP at
        # all is more efficient as no allreduce required in the end.
        # Note that if we see models that have large number of experts, we may
        # need to consider add TP back here.
        self.router = RowLinear(
            hidden_size,
            self.num_experts,
            bias=False,
            dtype=trt.
            float32,  # Routing is sensitive since it conditions what experts are used
            tp_group=None,
            tp_size=1,
            strict_dtype=True,
        )

        # Note we use horizontal fusion for gated activation to do the operation in one GEMM invocation
        #  The left matrix is a linear projection (no activation applied)
        #  The right matrix is the gating value (activation applied)
        # The naming convention is the inverse of GatedMLP, but the same as `tensorrt_llm/functional.py`
        expert_1_out_size = self.ffn_hidden_size * 2 if is_gated_activation(
            hidden_act) else self.ffn_hidden_size

        expert_1_shape = (self.experts_per_node, expert_1_out_size, hidden_size)
        expert_2_shape = (self.experts_per_node, hidden_size,
                          self.ffn_hidden_size)

        if quant_mode.is_weight_only():
            bytes_per_col_scale = 2 if quant_mode.is_int4_weight_only() else 1
            # We use a different shape here because the quantized weights have their own layout
            expert_1_shape = (self.experts_per_node, hidden_size,
                              expert_1_out_size // bytes_per_col_scale)
            expert_2_shape = (self.experts_per_node, self.ffn_hidden_size,
                              hidden_size // bytes_per_col_scale)

            self.experts_scale_1 = Parameter(shape=(self.experts_per_node,
                                                    expert_1_out_size),
                                             dtype=dtype)
            self.experts_scale_2 = Parameter(shape=(self.experts_per_node,
                                                    hidden_size),
                                             dtype=dtype)
        else:
            self.register_parameter('experts_scale_1', None)
            self.register_parameter('experts_scale_2', None)

        self.experts_weight_1 = Parameter(shape=expert_1_shape,
                                          dtype=self.weight_dtype)
        self.experts_weight_2 = Parameter(shape=expert_2_shape,
                                          dtype=self.weight_dtype)

        # Note: the bias uses dtype NOT weight_dtype, i.e. it is not quantized
        if bias:
            self.experts_bias_1 = Parameter(shape=(self.experts_per_node,
                                                   expert_1_out_size),
                                            dtype=dtype)
            self.experts_bias_2 = Parameter(shape=(self.experts_per_node,
                                                   hidden_size),
                                            dtype=dtype)
        else:
            self.register_parameter('experts_bias_1', None)
            self.register_parameter('experts_bias_2', None)

        ClsMLP = GatedMLP if is_gated_activation(self.hidden_act) else MLP

        # In OOTB mode, when ParallelismMode mode is TENSOR_PARALLEL, using MLP class to do TP settings
        # pass self.ffn_hidden_size to original size,
        # self.experts only inference in OOTB mode.
        if moe_config.tp_mode == MoeConfig.ParallelismMode.TENSOR_PARALLEL:
            ffn_hidden_size = self.ffn_hidden_size * self.tp_size
        else:
            tp_size = 1
            tp_group = None
            ffn_hidden_size = self.ffn_hidden_size
        self.experts = [
            ClsMLP(self.hidden_size, ffn_hidden_size,
                   non_gated_version(self.hidden_act), bias, dtype, tp_group,
                   tp_size, quant_mode) for _ in range(self.experts_per_node)
        ]

    def set_ootb_weight(self):
        for i, expert in enumerate(self.experts):
            is_gated_act = is_gated_activation(self.hidden_act)
            # Gated weight pack in expert1 weights
            # expert_weight_1
            experts_weight_1_raw = self.experts_weight_1.raw_value
            expert.fc.weight.value = experts_weight_1_raw[
                i, -self.ffn_hidden_size:, :]
            if is_gated_act:
                expert.gate.weight.value = experts_weight_1_raw[
                    i, :self.ffn_hidden_size, :]

            # expert_weight_2
            experts_weight_2_raw = self.experts_weight_2.raw_value
            expert.proj.weight.value = experts_weight_2_raw[i, :, :]

            has_bias = self.has_bias
            if has_bias:
                experts_bias_1_raw = self.experts_bias_1.raw_value
                expert.fc.bias.value = experts_bias_1_raw[
                    i, -self.ffn_hidden_size:]
                experts_bias_2_raw = self.experts_bias_2.raw_value
                expert.proj.bias.value = experts_bias_2_raw[i, :]
                if is_gated_act:
                    expert.gate.bias.value = experts_bias_1_raw[
                        i, :self.ffn_hidden_size]

    def forward(self, hidden_states, finished=None, lora_layer_params=None):
        assert lora_layer_params is None, "LoRA + MoE is not supported for the moment"
        routing_input = cast(hidden_states, trt.float32)
        routing = self.router(routing_input)

        if not default_net().plugin_config.moe_plugin:
            # Depending on the value of plugin_config.moe_plugin, weights must be assigned differently. Hence the need to do that in .forward().
            self.set_ootb_weight()

            if self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE:
                topk_values, topk_indices = topk(routing, self.top_k, dim=-1)
                topk_values = softmax(topk_values, -1)
            else:
                router_probs = softmax(routing, -1)
                topk_values, topk_indices = topk(router_probs,
                                                 self.top_k,
                                                 dim=-1)

            output = hidden_states * 0.0  # Create output space
            # Experts inference
            for i, expert in enumerate(self.experts):
                if self.tp_mode == MoeConfig.ParallelismMode.EXPERT_PARALLEL:
                    index = i + self.experts_per_node * self.tp_rank
                else:
                    index = i
                # inference expert
                out = expert(hidden_states)

                expert_mask = topk_indices == index
                expert_weights = cast(
                    sum(topk_values * cast(expert_mask, topk_values.dtype),
                        dim=-1,
                        keepdim=True), self.dtype)

                output += out * expert_weights
            if self.tp_size > 1 and self.tp_group is not None and self.moe_config.tp_mode == MoeConfig.ParallelismMode.EXPERT_PARALLEL:
                output = allreduce(output, self.tp_group)

        else:
            output = _moe_plugin(self.moe_config,
                                 hidden_states,
                                 routing,
                                 expert_weight_1=self.experts_weight_1,
                                 expert_weight_2=self.experts_weight_2,
                                 expert_bias_1=self.experts_bias_1,
                                 expert_bias_2=self.experts_bias_2,
                                 expert_scale_1=self.experts_scale_1,
                                 expert_scale_2=self.experts_scale_2,
                                 finished=finished,
                                 hidden_size=self.hidden_size,
                                 ffn_hidden_size=self.ffn_hidden_size,
                                 act_fn=self.hidden_act,
                                 dtype=self.dtype,
                                 weight_dtype=self.weight_dtype,
                                 quant_mode=self.quant_mode,
                                 tp_size=self.tp_size,
                                 tp_rank=self.tp_rank)

            if self.tp_size > 1 and self.tp_group is not None and self.moe_config.tp_mode != MoeConfig.ParallelismMode.NONE:
                output = allreduce(output, self.tp_group)

        return output


MOE = MixtureOfExperts
