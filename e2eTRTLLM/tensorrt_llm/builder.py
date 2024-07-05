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
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import tensorrt as trt

from ._common import _is_building, serialize_engine
from ._utils import (str_dtype_to_trt, support_strongly_type, to_dict,
                     to_json_file)
from .auto_parallel import auto_parallel
from .auto_parallel.config import AutoParallelConfig
from .graph_rewriting import optimize
from .logger import logger
from .lora_manager import LoraBuildConfig
from .models import PretrainedConfig, PretrainedModel
from .models.modeling_utils import optimize_model
from .network import Network, net_guard
from .plugin import PluginConfig
from .quantization import QuantAlgo, QuantMode
from .version import __version__


class BuilderConfig(object):

    def __init__(self, **kwargs):
        # intentionally use **kwargs, user should never call this ctor directly,
        # use Builder.create_builder_config() instead
        pass

    def _init(self, trt_builder_config, **kwargs):
        self._trt_builder_config = trt_builder_config
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @property
    def trt_builder_config(self) -> trt.IBuilderConfig:
        return self._trt_builder_config

    def to_dict(self) -> Dict:
        '''return a dict with keys
        {
            "builder_config": {
                # all key values set by the _init function
            },
            "plugin_config": {
                # the network plugin_config (if any) attached to this BuilderConfig object
                # inside the Builder.build_engine
            },
            "auto_parallel_config": {
                # the network auto_parallel_config (if any) attached to this BuilderConfig object
                # inside the Builder.build_engine
            }
        }
        '''
        config = {'builder_config': {}}
        for k in self.__dict__.keys():
            if k not in [
                    '_trt_builder_config', 'plugin_config',
                    'auto_parallel_config'
            ]:
                config['builder_config'][k] = self.__getattribute__(k)
        if hasattr(self, 'plugin_config'):
            assert isinstance(self.plugin_config, PluginConfig), \
                f"Found unexpected plugin_config object with type: {type(self.plugin_config)}"
            config['plugin_config'] = to_dict(self.plugin_config)
        return config


class Builder():

    _ALLOWED_PRECISIONS = ['float32', 'float16', 'bfloat16']

    def __init__(self):
        super().__init__()
        self._trt_builder = trt.Builder(logger.trt_logger)
        # TODO: Enable strongly_typed on by default in TRT 10.0
        self.strongly_typed = False

    @property
    def trt_builder(self) -> trt.Builder:
        return self._trt_builder

    def create_network(self) -> Network:
        explicit_batch_flag = 0
        # Explicit batch flag will be deprecated in TRT 10
        if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys(
        ):
            explicit_batch_flag = 1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        if support_strongly_type() and self.strongly_typed:
            return Network()._init(
                self.trt_builder.create_network(
                    explicit_batch_flag
                    | (1 << int(
                        trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))))
        else:
            return Network()._init(
                self.trt_builder.create_network(explicit_batch_flag))

    def create_builder_config(self,
                              precision: str,
                              timing_cache: Union[str, Path,
                                                  trt.ITimingCache] = None,
                              tensor_parallel: int = 1,
                              use_refit: bool = False,
                              int8: bool = False,
                              strongly_typed: bool = False,
                              opt_level: Optional[int] = None,
                              profiling_verbosity: str = "layer_names_only",
                              **kwargs) -> BuilderConfig:
        ''' @brief Create a builder config with given precisions and timing cache
            @param precision: one of allowed precisions, defined in Builder._ALLOWED_PRECISIONS
            @param timing_cache: a timing cache object or a path to a timing cache file
            @param tensor_parallel: number of GPUs used for tensor parallel
            @param kwargs: any other arguments users would like to attach to the config object as attributes
            @param refit: set to accelerate multi-gpu building, build engine for 1 gpu and refit for the others
            @param int8: whether to build with int8 enabled or not. Can't be used together with refit option
            @return: A BuilderConfig object, return None if failed
        '''
        if strongly_typed and not support_strongly_type():
            logger.warning(
                "TRT version does not support strongly_type. strongly_typed flag is ignored."
            )

        # In TRT 10.0, enable strongly_typed by default.
        self.strongly_typed = self.strongly_typed or (strongly_typed and
                                                      support_strongly_type())

        quant_mode = kwargs.get("quant_mode", QuantMode(0))
        if not strongly_typed and precision not in self._ALLOWED_PRECISIONS:
            logger.error(
                f"precision should be one of {self._ALLOWED_PRECISIONS}")

        if use_refit and int8:
            # TRT folds weights into Myelin graph because network contains int8 tensor or Q/DQ nodes
            # These folded weights can not be refitted
            logger.error("can't use refit and int8 mode at the same time")

        config = self.trt_builder.create_builder_config()
        if not self.strongly_typed:
            fp8 = quant_mode.has_fp8_qdq() or quant_mode.has_fp8_kv_cache()
            if precision == 'float16' or precision == trt.DataType.HALF:
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif precision == 'bfloat16' or precision == trt.DataType.BF16:
                config.set_flag(trt.BuilderFlag.BF16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)

            if fp8:
                config.set_flag(trt.BuilderFlag.FP8)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806,
                                   True)

        if use_refit:
            config.set_flag(trt.BuilderFlag.REFIT)

        if opt_level is not None:
            config.builder_optimization_level = opt_level

        # Set TRT Engine profiling verbosity
        if profiling_verbosity == "detailed":
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        elif profiling_verbosity == "none":
            config.profiling_verbosity = trt.ProfilingVerbosity.NONE
        else:
            config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY

        # set timing cache
        cache = None
        if timing_cache is not None:
            # use given cache
            if isinstance(timing_cache, trt.ITimingCache):
                cache = timing_cache
            # read cache from file
            elif isinstance(timing_cache,
                            (str, Path)) and os.path.exists(timing_cache):
                with open(timing_cache, "rb") as f:
                    cache = config.create_timing_cache(f.read())
            else:
                logger.warning(
                    "Invalid timing cache, using freshly created one")
        if cache is None:
            cache = config.create_timing_cache(b"")
        # When user does not given any existing cache, internally always created one
        # so the cache should never None here
        assert cache is not None and isinstance(cache, trt.ITimingCache)
        config.set_timing_cache(cache, ignore_mismatch=False)

        # set weight sparsity
        weight_sparsity = kwargs.get("weight_sparsity", False)
        if weight_sparsity:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        

        return BuilderConfig()._init(config,
                                     precision=precision,
                                     tensor_parallel=tensor_parallel,
                                     use_refit=use_refit,
                                     int8=int8,
                                     strongly_typed=self.strongly_typed,
                                     **kwargs)

    def _add_optimization_profile(self, network: Network,
                                  builder_config: BuilderConfig):
        assert isinstance(builder_config, BuilderConfig)
        assert isinstance(network, Network)
        input_tensors = network._inputs
        num_profiles = len(list(input_tensors.values())[0].profiles)
        for i in range(num_profiles):
            logger.debug(f'Adding optimization profile {i+1}/{num_profiles}')
            profile = self.trt_builder.create_optimization_profile()
            for input_name in input_tensors.keys():
                shape_profile = input_tensors[input_name].profiles[i]
                min_shape = [*shape_profile.min]
                opt_shape = [*shape_profile.opt]
                max_shape = [*shape_profile.max]
                if network._auto_parallel_config is not None:
                    io_shards = network._auto_parallel_config["io_shards"]
                    if input_name in io_shards:
                        shards = io_shards[input_name]
                        for dim, shard_num in shards.items():
                            min_shape[dim] = int(
                                math.floor(min_shape[dim] / shard_num))
                            opt_shape[dim] = int(
                                round(opt_shape[dim] / shard_num))
                            max_shape[dim] = int(
                                math.ceil(max_shape[dim] / shard_num))
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.debug(
                    f'{input_name}, min: {min_shape}, opt: {opt_shape}, max: {max_shape}, dimension names: {shape_profile.dimension_names}'
                )
            builder_config.trt_builder_config.add_optimization_profile(profile)
        assert self._validate_named_dimensions(
            network, builder_config
        ), "Validation of the tensor dimension ranges failed, please check the dimension ranges, find the offensive tensor and dimension name in above the error log"

    def _validate_named_dimensions(self, network: Network,
                                   builder_config) -> bool:
        '''
            For each profile, validate that the named dimensions of different input tensors in this profile all have same range.
            TRT will validate the same condition, validate it earlier to make sure the modeling in TensorRT-LLM are correct and
            makes the error msg more user friendly.
        '''
        valid = True
        for profile_idx in range(
                builder_config.trt_builder_config.num_optimization_profiles):
            dimension_to_range = {}
            for input_name, input_tensor in network._inputs.items():
                # it's legal that a Tensor does not have dim_range?
                if len(input_tensor.profiles) != 0:
                    profile = input_tensor.profiles[profile_idx]
                    for dim_idx, dim_name in enumerate(profile.dimension_names):
                        if dim_name not in dimension_to_range:
                            dimension_to_range[dim_name] = []
                        min, opt, max = profile.min[dim_idx], profile.opt[
                            dim_idx], profile.max[dim_idx]
                        dimension_to_range[dim_name].append(
                            (input_name, (min, opt, max)))
            for dim, ranges in dimension_to_range.items():
                unique_ranges = set([r[1] for r in ranges])
                logger.debug(
                    f"Validating dimension:{dim}, ranges for this dim are:{unique_ranges}"
                )
                if len(unique_ranges) != 1:
                    logger.error(
                        f"Found illegal dimension setting for profile {profile_idx}, dimension name is: {dim}"
                    )
                    logger.error(
                        f"Offensive tensors which have this dimension are:\n" +
                        "\n".join([f"{r[1]} {dim} {r[0]}" for r in ranges]))
                    valid = False
        return valid

    @_is_building
    def refit_engine(self, network: Network, engine_buffer) -> trt.IHostMemory:
        '''
            @brief: Refit one TensorRT engine using weights from the network,
                user should guarantee that the engine is built with REFIT flag, and the network has the same structure with the engine.
            @param engine_buffer: A serialized TensorRT engine.
            @param network: Network object.
            @return: A serialized TRT engine if refit successfully, None otherwise
        '''
        assert isinstance(network, Network)
        logger.info('Refit TRT engine')
        runtime = trt.Runtime(logger.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_buffer)

        tik = time.time()

        # Refit engine
        refitter = trt.Refitter(engine, logger.trt_logger)
        if network.named_parameters is not None:
            for name, param in network.named_parameters:
                if param._get_weights(
                ) is None or not refitter.set_named_weights(
                        name, param._get_weights()):
                    logger.error(f'Failed to refit weight: {name}')
                    return None
        else:
            logger.error(
                'Please set named parameters before building multiple engines.')
            return None

        if not refitter.refit_cuda_engine():
            logger.error('Failed to refit engine.')
            return None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of refitting {engine.name}: {t}')
        serialized_engine = engine.serialize()
        return serialized_engine

    @_is_building
    def build_engine(self, network: Network,
                     builder_config: BuilderConfig) -> trt.IHostMemory:
        '''
            @brief: Build one TensorRT engine from the network.
            @param network: Network object.
            @param builder_config: BuilderConfig object.
            @return: A serialized TRT engine.
        '''
        assert isinstance(network, Network)
        builder_config.plugin_config = network.plugin_config
        builder_config.auto_parallel_config = network.auto_parallel_config
        if builder_config.auto_parallel_config is not None:
            mapping = builder_config.auto_parallel_config["mapping"]
            builder_config.tensor_parallel = mapping.tp_size
            builder_config.pipeline_parallel = mapping.pp_size
        if builder_config.trt_builder_config.num_optimization_profiles == 0:
            self._add_optimization_profile(network, builder_config)
        engine = None

        # Rename weights
        if network.named_parameters is not None:
            for name, param in network.named_parameters:
                if param._get_weights() is None:
                    logger.info(
                        f"Parameter {name} {param.raw_value.shape} {param.raw_value.dtype} was not materialized to TRT network"
                    )
                    continue
                if not network.trt_network.set_weights_name(
                        param._get_weights(), name):
                    raise RuntimeError(f'Failed to set weight: {name}')

        network._fill_weights()
        # Build engine
        logger.info(f'Build TensorRT engine {network.trt_network.name}')
        tik = time.time()
        engine = self.trt_builder.build_serialized_network(
            network.trt_network, builder_config.trt_builder_config)
        if engine is None:
            logger.error('Engine building failed, please check the error log.')
            return None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of building {network.trt_network.name}: {t}')

        return engine

    @staticmethod
    def save_timing_cache(builder_config: BuilderConfig, out_path: str) -> bool:
        '''Serialize timing cache of given builder config to file specified by out_path
            return True if the cache is successfully serialized, False otherwise
        '''
        cache = builder_config.trt_builder_config.get_timing_cache()
        if cache is None:
            logger.warning(
                'No timing cache found in the given builder config, skip saving.'
            )
            return False
        with cache.serialize() as buffer:
            with open(out_path, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f)
        logger.info(f'Timing cache serialized to {out_path}')
        return True

    @staticmethod
    def save_config(builder_config: BuilderConfig, config_path: str):
        config = builder_config.to_dict()
        to_json_file(config, config_path)
        logger.info(f'Config saved to {config_path}.')


@dataclass
class BuildConfig:
    max_input_len: int = 256
    max_output_len: int = 256
    max_batch_size: int = 8
    max_beam_width: int = 1
    max_num_tokens: Optional[int] = None
    opt_num_tokens: Optional[int] = None
    max_prompt_embedding_table_size: int = 0
    gather_context_logits: int = False
    gather_generation_logits: int = False
    strongly_typed: bool = False
    builder_opt: Optional[int] = None
    profiling_verbosity: str = 'layer_names_only'
    enable_debug_output: bool = False
    max_draft_len: int = 0
    use_refit: bool = False
    input_timing_cache: str = None
    output_timing_cache: str = None
    lora_config: LoraBuildConfig = LoraBuildConfig()
    auto_parallel_config: AutoParallelConfig = AutoParallelConfig()
    weight_sparsity: bool = False
    plugin_config: PluginConfig = PluginConfig()
    max_encoder_input_len: int = 1  # for enc-dec DecoderModel
    use_fused_mlp: bool = False

    @classmethod
    def from_dict(cls, config, plugin_config=None):
        max_input_len = config.pop('max_input_len')
        max_output_len = config.pop('max_output_len')
        max_batch_size = config.pop('max_batch_size')
        max_beam_width = config.pop('max_beam_width')
        max_num_tokens = config.pop('max_num_tokens')
        opt_num_tokens = config.pop('opt_num_tokens')
        max_prompt_embedding_table_size = config.pop(
            'max_prompt_embedding_table_size', 0)
        gather_context_logits = config.pop('gather_context_logits', False)
        gather_generation_logits = config.pop('gather_generation_logits', False)
        strongly_typed = config.pop('strongly_typed', False)
        builder_opt = config.pop('builder_opt', None)
        weight_sparsity = config.pop('weight_sparsity', False)
        profiling_verbosity = config.pop('profiling_verbosity',
                                         'layer_names_only')
        enable_debug_output = config.pop('enable_debug_output', False)
        max_draft_len = config.pop('max_draft_len', 0)
        use_refit = config.pop('use_refit', False)
        input_timing_cache = config.pop('input_timing_cache', None)
        output_timing_cache = config.pop('output_timing_cache', None)
        lora_config = LoraBuildConfig.from_dict(config.get('lora_config', {}))
        auto_parallel_config = AutoParallelConfig.from_dict(
            config.get('auto_parallel_config', {}))
        max_encoder_input_len = config.pop('max_encoder_input_len', 1024)

        if plugin_config is None:
            plugin_config = PluginConfig()
        if "plugin_config" in config.keys():
            plugin_config.update_from_dict(config["plugin_config"])
        return cls(
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            strongly_typed=strongly_typed,
            builder_opt=builder_opt,
            profiling_verbosity=profiling_verbosity,
            enable_debug_output=enable_debug_output,
            max_draft_len=max_draft_len,
            use_refit=use_refit,
            input_timing_cache=input_timing_cache,
            output_timing_cache=output_timing_cache,
            lora_config=lora_config,
            auto_parallel_config=auto_parallel_config,
            max_encoder_input_len=max_encoder_input_len,
            weight_sparsity=weight_sparsity,
            plugin_config=plugin_config)

    @classmethod
    def from_json_file(cls, config_file, plugin_config=None):
        with open(config_file) as f:
            config = json.load(f)
            return BuildConfig.from_dict(config, plugin_config=plugin_config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        plugin_config = output.pop('plugin_config')
        plugin_config_dict = copy.deepcopy(plugin_config.__dict__)
        output['plugin_config'] = plugin_config_dict
        output['lora_config'] = output['lora_config'].to_dict()
        output['auto_parallel_config'] = output['auto_parallel_config'].to_dict(
        )
        return output


class EngineConfig:

    def __init__(self, pretrained_config: 'PretrainedConfig',
                 build_config: 'BuildConfig', version: str):
        self.pretrained_config = pretrained_config
        self.build_config = build_config
        self.version = version

    @classmethod
    def from_json_file(cls, config_file):
        with open(config_file) as f:
            config = json.load(f)
            return cls(PretrainedConfig.from_dict(config['pretrained_config']),
                       BuildConfig.from_dict(config['build_config']),
                       config['version'])

    def to_dict(self):
        return {
            'version': self.version,
            'pretrained_config': self.pretrained_config.to_dict(),
            'build_config': self.build_config.to_dict(),
        }


class Engine:

    def __init__(self, config: EngineConfig, engine: trt.IHostMemory):
        self.config = config
        self.engine = engine

    def save(self, engine_dir: str):
        os.makedirs(engine_dir, exist_ok=True)
        lora_config = self.config.build_config.lora_config
        lora_dirs = lora_config.lora_dir
        root_lora_dir = os.path.join(engine_dir, 'lora')
        if len(lora_dirs) > 0:
            os.makedirs(root_lora_dir, exist_ok=True)
            for index, lora_dir in enumerate(lora_dirs):
                if lora_config.lora_ckpt_source == 'hf':
                    target_lora_dir = f"{root_lora_dir}/{index}"
                    os.makedirs(target_lora_dir, exist_ok=True)
                    shutil.copy2(os.path.join(lora_dir, 'adapter_config.json'),
                                 target_lora_dir)
                    shutil.copy2(os.path.join(lora_dir, 'adapter_model.bin'),
                                 target_lora_dir)
                    lora_config.lora_dir[index] = f"lora/{index}"
                elif lora_config.lora_ckpt_source == 'nemo':
                    target_lora_file = f"{root_lora_dir}/{index}.nemo"
                    shutil.copyfile(lora_dir, target_lora_file)
                    lora_config.lora_dir[index] = f"lora/{index}.nemo"
        else:
            if os.path.exists(root_lora_dir) and os.path.isdir(root_lora_dir):
                shutil.rmtree(root_lora_dir)
        if self.config.pretrained_config.mapping.rank == 0:
            with open(os.path.join(engine_dir, 'config.json'),
                      "w",
                      encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=4)
        serialize_engine(
            self.engine,
            os.path.join(
                engine_dir,
                f'rank{self.config.pretrained_config.mapping.rank}.engine'))

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int = 0):
        with open(os.path.join(engine_dir, f'rank{rank}.engine'), 'rb') as f:
            engine_buffer = f.read()

        config = EngineConfig.from_json_file(
            os.path.join(engine_dir, 'config.json'))
        config.pretrained_config.set_rank(rank)

        return cls(config, engine_buffer)


def get_engine_version(engine_dir: str) -> Union[None, str]:
    engine_dir = Path(engine_dir)
    config_path = engine_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'version' not in config:
        return None

    return config['version']


def build(model: PretrainedModel, build_config: BuildConfig) -> Engine:
    '''Build engine from given model and optimization options specified in the build_config
       WARNING: this function may change the given \p model object state in some optimization passes
       to avoid cloning a model since normally the LLM models consumes large memory.
       Create a new fresh model object if you need to build with different options.

    '''
    build_config = copy.deepcopy(
        build_config)  # avoid changing the input config
    if model.config.quantization.quant_algo == QuantAlgo.FP8 or \
        model.config.quantization.kv_cache_quant_algo == QuantAlgo.FP8:
        build_config.strongly_typed = True

    if hasattr(model.config, 'max_medusa_token_len'):
        build_config.max_draft_len = model.config.max_medusa_token_len

    use_auto_parallel = build_config.auto_parallel_config.enabled

    if model.config.architecture not in ["EncoderModel", "DecoderModel"]:
        model = optimize_model(
            model,
            use_fused_mlp=(build_config.use_fused_mlp
                           and not use_auto_parallel),
            use_prompt_tuning=(build_config.max_prompt_embedding_table_size >
                               0))

    if build_config.plugin_config.lora_plugin is not None:
        model.use_lora(build_config.lora_config)
        model = optimize_model(
            model,
            use_lora=True,
            max_lora_rank=build_config.lora_config.max_lora_rank,
        )

    builder = Builder()
    builder_config = builder.create_builder_config(
        precision=model.config.dtype,
        use_refit=build_config.use_refit,
        timing_cache=build_config.input_timing_cache,
        int8=(model.config.quant_mode.has_act_or_weight_quant()
              and not model.config.quant_mode.has_per_group_scaling())
        or model.config.quant_mode.has_int8_kv_cache(),
        strongly_typed=build_config.strongly_typed,
        opt_level=build_config.builder_opt,
        profiling_verbosity=build_config.profiling_verbosity,
        quant_mode=model.config.quant_mode,
        weight_sparsity=build_config.weight_sparsity,
    )

    network = builder.create_network()
    network.plugin_config = build_config.plugin_config

    use_weight_only = model.config.quant_mode.is_weight_only()
    per_group = model.config.quant_mode.has_per_group_scaling()
    use_smooth_quant = model.config.quant_mode.has_act_and_weight_quant()
    disable_weight_only_quant_plugin = model.config.disable_weight_only_quant_plugin if hasattr(
        model.config, 'disable_weight_only_quant_plugin') else False

    if use_weight_only and not disable_weight_only_quant_plugin:
        if per_group:
            network.plugin_config.set_plugin(
                "weight_only_groupwise_quant_matmul_plugin", model.config.dtype)
        else:
            network.plugin_config.set_plugin("weight_only_quant_matmul_plugin",
                                             model.config.dtype)
    if use_smooth_quant and model.config.quantization.use_plugin_sq:
        network.plugin_config.set_smooth_quant_plugins()
    if network.plugin_config.use_paged_context_fmha:
        if (model.config.quant_mode.has_fp8_kv_cache()
                and not model.config.quant_mode.has_fp8_qdq()):
            raise RuntimeError(
                "FP8 Paged Context FMHA only works with fp8 quantization workflow currently."
            )
        if model.config.quant_mode.has_int8_kv_cache():
            raise RuntimeError(
                "Paged Context FMHA doesn't work with int8 kv cache currently.")
    nccl_plugin = model.config.dtype if model.config.mapping.world_size > 1 else None
    network.plugin_config.set_nccl_plugin(
        nccl_plugin, network.plugin_config.use_custom_all_reduce)

    use_auto_parallel = build_config.auto_parallel_config.enabled
    model = optimize_model(model, use_unfused_qkv_gemm=use_auto_parallel)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(model.named_parameters())

        # Forward
        prepare_input_args = {
            "max_batch_size": build_config.max_batch_size,
            "max_input_len": build_config.max_input_len,
            "max_seq_len":
            build_config.max_input_len + build_config.max_output_len,
            "use_cache": True,
            "max_beam_width": build_config.max_beam_width,
            "max_num_tokens": build_config.max_num_tokens,
            "opt_num_tokens": build_config.opt_num_tokens,
            "prompt_embedding_table_size":
            build_config.max_prompt_embedding_table_size,
            "max_draft_len": build_config.max_draft_len,
            "gather_context_logits": build_config.gather_context_logits,
            "gather_generation_logits": build_config.gather_generation_logits,
            "lora_target_modules": build_config.lora_config.lora_target_modules
        }

        if model.config.architecture == "DecoderModel":
            prepare_input_args["max_seq_len"] = build_config.max_output_len
            prepare_input_args[
                "max_decoder_input_len"] = build_config.max_input_len
            prepare_input_args[
                "max_encoder_input_len"] = build_config.max_encoder_input_len

        inputs = model.prepare_inputs(**prepare_input_args)
        model(**inputs)

        if build_config.enable_debug_output:
            for k, v in model.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(model.config.dtype))

    if model.config.architecture != "DecoderModel":
        optimize(network)

    if use_auto_parallel:
        config = build_config.auto_parallel_config
        config.builder_flags = builder_config.trt_builder_config.flags
        sharded_networks = auto_parallel(network, config)
        network = sharded_networks[model.config.mapping.rank]
        if not build_config.auto_parallel_config.debug_mode:
            mapping = network.auto_parallel_config["mapping"]
            model.config.mapping = mapping

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    engine_config = EngineConfig(model.config, build_config, __version__)

    if build_config.output_timing_cache is not None and model.config.mapping.rank == 0:
        ok = builder.save_timing_cache(builder_config,
                                       build_config.output_timing_cache)
        assert ok, "Failed to save timing cache."

    return Engine(engine_config, engine)
