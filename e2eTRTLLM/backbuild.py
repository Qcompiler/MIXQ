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
        out.trt_tensor.name = 'out'


    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    config_path = Path(args.output_dir) / 'config.json'
    builder.save_config(builder_config, str(config_path))
    return engine




from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.functional import constant
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
    mat1 = torch.randn(shape1, dtype=torch.float16)

    shape2 = (args.out_features, args.in_features)
    mat2 = torch.randint(0, 2, shape2, dtype=torch.int8)
    shape_scale_a = (args.batch_size * args.seq_len, 1)  
    scale_a_torch = torch.ones(shape_scale_a, dtype=torch.float16) 
    # scale_a_torch *= torch.randint(1,
    #                                 10,
    #                                 shape_scale_a,
    #                                 dtype=torch.float16)
    shape_scale_b = (1, args.out_features)  
    scale_b_torch = torch.ones(shape_scale_b, dtype=torch.float16) 
    # scale_b_torch *= torch.randint(1,
    #                                 10,
    #                                 shape_scale_b,
    #                                 dtype=torch.float16)
        
    # # Infer engine


    with tensorrt_llm.net_guard(net):
        network = tensorrt_llm.default_trtnet()
        # Init TensorRT-LLM tensor for mat1
        x = Tensor(name='x',
                    shape=mat1.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))


        weights = constant(torch_to_numpy(mat2))
        # Init TensorRT-LLM tensor for per channel scaling
        scale_weight = constant(torch_to_numpy(scale_b_torch))
        
        scale_a = Tensor(
            name='scale_a',
            shape=scale_a_torch.shape,
            dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
        # Init TensorRT-LLM tensor for per channel scaling

        # Get output tensor for SQ gemm
        output = w8awgemm(args.batch_size*args.seq_len, args.out_features, args.in_features, 
                          [x.trt_tensor, weights.trt_tensor, 
                           scale_a.trt_tensor, scale_weight.trt_tensor]).trt_tensor
        output.name = 'output'
        network.mark_output(output)
        output.dtype = tensorrt_llm._utils.str_dtype_to_trt("int32")
        
    build_engine = EngineFromNetwork(
        (builder.trt_builder, net.trt_network),
        config=CreateConfig(
            int8=True,
            fp16=False,
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))
    

    # Infer engine
    with TrtRunner(build_engine) as runner:
        outputs = runner.infer(
            feed_dict={
                'x': mat1.numpy(),
                'y': mat2.numpy(),
                'scale_a': scale_a_torch.numpy(),
                'scale_b': scale_b_torch.numpy()
            })
    print(outputs['output'])
    #exit(0)

    # mat1 = mat1.reshape((-1,mat1.shape[-1]))



    # tmp = torch.mm(mat1.to(torch.float32),mat2.to(torch.float32).T)


    # tmp2 = torch.mm(scale_a_torch, scale_b_torch).to(torch.float32)
    # out = torch.mul( tmp, tmp2)
    # print(out)
    # exit(0)

    # builder = Builder()
    # cache = None
    # builder_config = builder.create_builder_config(
    #     name=LAYER_NAME,
    #     precision= 'fp16',
    #     timing_cache=args.timing_cache if cache is None else cache,
    #     profiling_verbosity=args.profiling_verbosity)

    # engine_name = get_engine_name(args.in_features, args.dtype)
    # engine = build_engine(builder, builder_config, engine_name, args)
    # assert engine is not None

    # engine_path = output_dir / engine_name
    # logger.info(f'Serializing engine to {str(engine_path)}...')
    # tik = time.time()
    # with engine_path.open('wb') as f:
    #     f.write(engine)
    # tok = time.time()
    # t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    # logger.info(f'Engine serialized. Total time: {t}')

    # ok = builder.save_timing_cache(builder_config,
    #                                Path(args.output_dir) / "model.cache")
    # assert ok, "Failed to save timing cache."


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
