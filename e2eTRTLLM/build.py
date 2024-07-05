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
import argparse
import math
import time
from pathlib import Path

import tensorrt as trt
from plugin import LAYER_NAME, MixQLinear, get_engine_name

import tensorrt_llm
from tensorrt_llm.builder import Builder, BuilderConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

from plugin import w8awgemm
def build_engine(builder: Builder, builder_config: BuilderConfig,
                 engine_name: str, args: argparse.Namespace) -> trt.IHostMemory:
    '''

    @brief: Build a TensorRT engine.
    @param args: The cmd line arguments.
    @return: The built or refitted engine.
    '''

    # Initialize Module
     
    layer = MixQLinear(args.in_features, args.out_features)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    network.plugin_config.to_legacy_setting()
    with net_guard(network):
        # Prepare
        inputs = layer.prepare_inputs(args.batch_size, args.in_features, args.out_features)
        # Forward
        logger.debug(f'model inputs: {inputs}')
        out = layer(*inputs)


        network._mark_output(out,'out',trt.float16)
        out.dtype = tensorrt_llm._utils.str_dtype_to_trt("float16")

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    config_path = Path(args.output_dir) / 'config.json'
    builder.save_config(builder_config, str(config_path))
    return engine




import torch
def torch_quant(mat_float,shape,trans=False):
    mat_float_shape = mat_float.shape
    mat_float = mat_float.reshape(-1,mat_float.shape[-1])

    #print(mat_float.shape)
    #print(torch.max(torch.abs(mat_float), dim=dim)[0].unsqueeze(1).shape)
    scale_torch =   (torch.max(torch.abs(mat_float), dim=1)[0].unsqueeze(1) / (
                                127)).to(torch.float16).reshape(shape)
        
    #print(shape)
    #print(scale_torch.shape)

    tmp = mat_float
    if trans:
        tmp /= scale_torch.T
    else:
        tmp /= scale_torch
    mat = tmp.round().to(torch.int8)
    return mat.reshape(mat_float_shape), scale_torch

from tensorrt_llm.functional import constant
from tensorrt_llm._utils import torch_to_numpy
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from tensorrt_llm import Tensor
def build(args):
    tensorrt_llm.logger.set_level(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = tensorrt_llm.Builder()
    net = builder.create_network()

    args.seq_len = 1;
    import torch
    shape1 = (args.batch_size,args.seq_len, args.in_features)
    mat1_float = torch.rand(shape1, dtype=torch.float16)  

    shape2 = (args.out_features, args.in_features)
    mat2_float = torch.rand(shape2, dtype=torch.float16)

    shape_scale_a = (args.batch_size * args.seq_len, 1)
    shape_scale_b = (1, args.out_features)  

    
 

    mat2, scale_b_torch  = torch_quant(torch.clone(mat2_float),shape_scale_b, trans = 1)
    
    # [ ]12288 * 4096 
        
    # # Infer engine


    with tensorrt_llm.net_guard(net):
        network = tensorrt_llm.default_trtnet()
        # Init TensorRT-LLM tensor for mat1
        x = Tensor(name='x',
                    shape=mat1_float.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
        # Init TensorRT-LLM tensor for mat2
        #weight = constant(torch_to_numpy(mat2))
        weight = Tensor(name='weight',
                    shape=mat2.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))

 
        #scale_b = constant(torch_to_numpy(scale_b_torch))
        # Init TensorRT-LLM tensor for per channel scaling
        scale_b = Tensor(
            name='scale_b',
            shape=scale_b_torch.shape,
            dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
        # Get output tensor for SQ gemm
        output = w8awgemm(args.batch_size*args.seq_len, args.out_features, args.in_features, 
                          [x.trt_tensor, weight.trt_tensor, 
                            scale_b.trt_tensor]).trt_tensor
        output.name = 'output'
        network.mark_output(output)
        output.dtype = tensorrt_llm._utils.str_dtype_to_trt("float16")
        
    build_engine = EngineFromNetwork(
        (builder.trt_builder, net.trt_network),
        config=CreateConfig(
            int8=True,
            fp16=True,
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 101712384}))
    
    
    # Infer engine
    with TrtRunner(build_engine) as runner:
        
        outputs = runner.infer(
            feed_dict={
                'x': mat1_float.numpy(),
                'weight': mat2.numpy(),
                'scale_b': scale_b_torch.numpy()
            })
 
    mat1_float = mat1_float.reshape((-1,mat1_float.shape[-1]))

 
    grand = torch.mm(mat1_float, mat2_float.T)

    out = torch.as_tensor(outputs['output'],dtype=torch.float16).reshape(grand.shape)
    print(out)
    print(grand)

 
    exit(0)

    builder = Builder()
    cache = None
    builder_config = builder.create_builder_config(
        name=LAYER_NAME,
        precision= 'float16',
        int8 = True,
        fp16 = True,
        timing_cache=args.timing_cache if cache is None else cache,
        profiling_verbosity=args.profiling_verbosity)

    engine_name = get_engine_name(args.in_features, args.dtype)
    engine = build_engine(builder, builder_config, engine_name, args)
    assert engine is not None

    engine_path = output_dir / engine_name
    logger.info(f'Serializing engine to {str(engine_path)}...')
    tik = time.time()
    with engine_path.open('wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')

    ok = builder.save_timing_cache(builder_config,
                                   Path(args.output_dir) / "model.cache")
    assert ok, "Failed to save timing cache."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_features', type=int, default=4096)
    parser.add_argument('--out_features', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help='The path of to read timing cache from, will be ignored '
        'if the file does not exist')
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='The path to save the serialized engine files, timing cache '
        'file and model configs')
    args = parser.parse_args()

    logger.set_level(args.log_level)
    logger.info('Parameters'.center(40, '='))
    for k, v in vars(args).items():
        logger.info(f' - {k.ljust(15, ".")}: {v}')
    logger.info(''.center(40, '='))

    tik = time.time()
    logger.info('Build TensorRT engine.')
    build(args)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building TRT engine: {t}')
