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
import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor, _create_tensor,_add_plugin_info
from tensorrt_llm.module import Module
from tensorrt_llm._common import default_net, default_trtnet

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'
LAYER_NAME = 'MixQLayer'
FMHA_KERNEL_BLOCK_SIZE = 128


def _load_triton_plugin_lib():
    triton_plugin_dir = Path(__file__).parent.absolute()
    plugin_lib = triton_plugin_dir / 'build/libtrt_llm_custom_plugins.so'
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT-LLM Triton Plugin is unavailable')
    handle.initOpenAiTritonPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initOpenAiTritonPlugins.restype = ctypes.c_bool
    assert handle.initOpenAiTritonPlugins(
        None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))

print("loading plugging ")
_load_triton_plugin_lib()
print("done! ")


def flash_attention_op(num_heads: int, head_size: int, softmax_scale: float,
                       inputs: List[trt.ITensor]) -> Tensor:
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixQ', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    pfc = trt.PluginFieldCollection([
        trt.PluginField("num_heads", np.array([num_heads], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([head_size], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("softmax_scale", np.array([softmax_scale], np.float32),
                        trt.PluginFieldType.FLOAT32),
        trt.PluginField("type_id", np.array([int(inputs[0].dtype)], np.int32),
                        trt.PluginFieldType.INT32)
    ])
    plugin = plugin_creator.create_plugin("flash_attention", pfc)
    layer = default_trtnet().add_plugin_v2(inputs, plugin)
    return _create_tensor(layer.get_output(0), layer)


def w8awgemm(m: int, n: int, k: int, 
                       inputs: List[trt.ITensor]) -> Tensor:
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixQ', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    pfc = trt.PluginFieldCollection([
        trt.PluginField("m", np.array([m], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("n", np.array([n], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("k", np.array([k], np.int32),
                        trt.PluginFieldType.INT32),
    ])

     
    plugin = plugin_creator.create_plugin("flash_attention", pfc)
    layer = default_trtnet().add_plugin_v2(inputs, plugin)


    _add_plugin_info(layer, plugin_creator, "flash_attention", pfc)
    if not default_net().strongly_typed:
        layer.get_input(1).set_dynamic_range(-127, 127)
 
    

    return _create_tensor(layer.get_output(0), layer)


def mixgemm(m: int, n: int, k: int, 
                       inputs: List[trt.ITensor]) -> Tensor:
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixQ', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    pfc = trt.PluginFieldCollection([
        trt.PluginField("m", np.array([m], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("n", np.array([n], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("k", np.array([k], np.int32),
                        trt.PluginFieldType.INT32),
    ])

     
    plugin = plugin_creator.create_plugin("tsinghua_mixQ", pfc)
    layer = default_trtnet().add_plugin_v2(inputs, plugin)


    _add_plugin_info(layer, plugin_creator, "tsinghua_mixQ", pfc)
    if not default_net().strongly_typed:
        layer.get_input(1).set_dynamic_range(-127, 127)
 
    

    return _create_tensor(layer.get_output(0), layer)

from tensorrt_llm.parameter import Parameter

from tensorrt_llm.quantization.mode import QuantMode
from tensorrt_llm.functional import allreduce

class MixQLinear(Module):

    def __init__(self, in_features: int, out_features: int,
                 bias=False, 
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 quant_mode=QuantMode.use_mix_precision()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size
        self.dtype =  str_dtype_to_trt('float16')
        self.weight = Parameter(shape=(self.out_features, 
                                       self.in_features // 2),
                                dtype=trt.float16)
        self.fp_weight = Parameter(shape=(self.out_features, 
                                       128),
                                dtype=trt.float16)
        self.fp_ind = Parameter(shape=(128 * 2, ),
                                dtype=trt.float16)
        

        self.qweight = Parameter(shape=(self.in_features , 
                                       self.out_features // 2),
                                dtype=trt.float16)    
        # self.qzeros = Parameter(shape=(self.in_features // 128, 
        #                                self.out_features // 4),
        #                         dtype=trt.float16)          
        self.weights_scaling_factor = Parameter(shape=(self.out_features, 1),
                                dtype=trt.float16)               
        # self.register_buffer('qweight', torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        # self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        # self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev))         
        
        scale_shape = (self.out_features, )
        self.weights_scaling_factor = Parameter(shape=scale_shape,
                                            dtype=trt.float16)




        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output
        if bias:
            self.bias = Parameter(shape=(self.out_features, ), 
                                  dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self,  A : Tensor, lora_runtime_params=None):
        # A activation
        #print('---------------------forward------------------')
        #print(A.shape)
        x = mixgemm(A.shape[0],self.out_features,
                       self.in_features,
                       inputs= [A.trt_tensor,
                               self.weight.value.trt_tensor, 
                               self.weights_scaling_factor.value.trt_tensor,
                                self.fp_weight.value.trt_tensor, #新增fp weight 
                                self.fp_ind.value.trt_tensor, #新增fp ind 
                                self.qweight.value.trt_tensor, #新增 int4 weight int32
                                self.weights_scaling_factor.value.trt_tensor, #新增   scaling_factors  half
                          
                
                               ])

        #x.mark_output('out', self.dtype)
        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)
        assert self.bias is None
         
        return x

    def prepare_inputs(self, max_batch_size: int, 
                       in_features: int, 
                       out_features: int,) -> List[Tensor]:

      
        raise NotImplementedError("!")
        dynamic_shape = [256, 1, in_features ]
        A = Tensor(name='A',
                   dtype=str_dtype_to_trt('float16'),
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [256]),
                       ('seq_len', [1]),
                       ('in_features', [in_features]),
                   ]
                   )
                   )

        return [A, ]
    
class FmhaLayer(Module):

    def __init__(self, num_heads: int, head_size: int, softmax_scale: float,
                 dtype: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.dtype = str_dtype_to_trt(dtype)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        inputs = [Q, K, V]
        out = flash_attention_op(num_heads=self.num_heads,
                                 head_size=self.head_size,
                                 softmax_scale=self.softmax_scale,
                                 inputs=[p.trt_tensor for p in inputs])
        out.mark_output('out', self.dtype)
        return out

    def prepare_inputs(self, max_batch_size: int, max_len: int) -> List[Tensor]:
        '''

        @brief: Prepare inputs Tensors for the model, the given sizes are used to
            determine the ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        '''

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        max_len_range = [1, (max_len + 1) // 2, max_len]

        dynamic_shape = [-1, self.num_heads, -1, self.head_size]
        Q = Tensor(name='Q',
                   dtype=self.dtype,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        K = Tensor(name='K',
                   dtype=self.dtype,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        V = Tensor(name='V',
                   dtype=self.dtype,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        return [Q, K, V]


def get_engine_name(head_size, dtype):
    return f'{LAYER_NAME}_{FMHA_KERNEL_BLOCK_SIZE}_d{head_size}_{dtype}.engine'
