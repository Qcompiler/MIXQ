# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Common utils for the ModelConfig."""

import dataclasses
import math
from typing import Dict, Union, get_args, get_origin

import numpy as np
import torch

from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_INT8_Mix,
    QUANTIZATION_W4A8_AWQ,
    DecoderLayerConfig,
    LayernormConfig,
    LinearConfig,
    ModelConfig,
    QKVConfig,
)

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype("V2", metadata={"dtype": "bfloat16"})


def _numpy_to_torch(x):
    """Convert numpy array to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x

    if x.dtype != np_bfloat16:
        return torch.tensor(x)
    return torch.tensor(x.view(np.int16)).view(torch.bfloat16)


def model_config_to_dict(model_config: ModelConfig) -> dict:
    """Converts the instance to a python dict."""
    assert model_config is not None, "model_config is None"
    return dataclasses.asdict(model_config)


def split_config_and_weights(config, weights: Dict[str, torch.tensor], prefix: str = "transformer"):
    """Util function to split the weights or any torch.Tensor in nested config to weights.

    A weight id starts with transformers or lm_head will also be generated to link the original key to the weights dict.
    The weights in the weights dict are contiguous.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            if k == "lm_head":
                # lm_head is not part of the transformer.
                array_key = k
            else:
                array_key = f"{prefix}.{k}"
            if isinstance(v, torch.Tensor):
                weights[array_key] = v
                config[k] = f"{array_key}"
            else:
                split_config_and_weights(v, weights, array_key)
    elif isinstance(config, list):
        for i, v in enumerate(config):
            array_key = f"{prefix}.{i}"
            if isinstance(v, torch.Tensor):
                weights[array_key] = v
                config[i] = f"{array_key}"
            else:
                split_config_and_weights(v, weights, array_key)


def _unified_weights_key(k: str) -> str:
    """Try to unify the weights dict key between old npz and the new safetensors format."""
    prefixes = ["transformer.", "_np:"]
    for prefix in prefixes:
        if k.startswith(prefix):
            k = k[len(prefix) :]

    k = k.replace("final_layernorm", "ln_f")

    return k.replace(":", ".")


def _restore_model_config(model_config, weights: Dict[str, Union[np.ndarray, torch.Tensor]]):
    def _is_tensor_key(k):
        return isinstance(k, str) and _unified_weights_key(k) in weights

    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if _is_tensor_key(v):
                model_config[k] = _numpy_to_torch(weights[_unified_weights_key(v)])
            else:
                _restore_model_config(v, weights)
    if isinstance(model_config, list):
        for i, v in enumerate(model_config):
            if _is_tensor_key(v):
                model_config[i] = _numpy_to_torch(weights[_unified_weights_key(v)])
            else:
                _restore_model_config(v, weights)


def restore_model_config(model_config, weights: Dict[str, Union[np.ndarray, torch.Tensor]]):
    """Recursively restores the model_config from json and loads np.ndarray or torch.Tensor weights from weights."""
    unified_key_weights = {}
    for k, v in weights.items():
        unified_key_weights[_unified_weights_key(k)] = v

    _restore_model_config(model_config, unified_key_weights)


def _from_dict(class_type, data):
    """Helper function to load the data as a class_type. class_type must be a dataclass."""
    if data is None:
        return None

    if get_origin(class_type) == Union:
        # Handle QKV
        if all([key in data for key in ["q", "k", "v"]]):
            # splitted qkv case
            class_type = QKVConfig
        else:
            # merged qkv case
            assert "linear_type" in data, f"{data} is not a valid LinearConfig"
            class_type = LinearConfig

    if dataclasses.is_dataclass(class_type):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(class_type)}
        fields_map = {}
        for k, v in data.items():
            if k in fieldtypes:
                # We only handle keys available in the fields.
                # Deprecated fields in the checkpoint will be ignored.
                fields_map[k] = _from_dict(fieldtypes[k], v)
        return class_type(**fields_map)
    elif get_origin(class_type) == list and dataclasses.is_dataclass(get_args(class_type)[0]):
        list_value = []
        for child in data:
            child_class_type = get_args(class_type)[0]
            list_value.append(_from_dict(child_class_type, child))
        return list_value
    else:
        return data


def model_config_from_dict(d: dict) -> ModelConfig:
    """Load a dict to a `ModelConfig` instance."""
    config_type = ModelConfig

    config_type_map = {}
    for t in [ModelConfig, DecoderLayerConfig, LayernormConfig, LinearConfig]:
        config_type_map[t.__name__] = t

    if "__name__" in d:
        config_name = d.pop("__name__")
        try:
            config_type = config_type_map[config_name]
        except Exception as e:
            raise NotImplementedError(f"{config_name} not supported") from e

    return _from_dict(config_type, d)


def pad_weights(weights, tp_size):
    """Returns the padded weights to tp_size."""
    assert len(weights.shape) > 1

    def _pad_size(original_size, tp_size):
        return int(math.ceil(original_size / tp_size) * tp_size)

    original_size = weights.shape[0]
    padded_size = _pad_size(original_size, tp_size)

    if original_size != padded_size:
        pad_width = padded_size - original_size
        return torch.nn.functional.pad(weights, (0, 0, 0, pad_width), "constant", value=0)
    return weights

def awq_from_linear( linear, w_bit = 4, group_size = 128, scales=None, zeros=None):
 
        awq_linear_w_bit = w_bit
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales
        
        awq_linear_scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear_bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear_w_bit

        intweight = []
        for idx in range(linear.weight.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + 
                                          scale_zeros[idx // group_size]) / 
                                          awq_linear_scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear_w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear_w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear_w_bit)

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear_w_bit), dtype=torch.int32, device=zeros.device)
        
        for col in range(zeros.shape[1] // pack_num):
            if awq_linear_w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear_w_bit)


        return qweight, qzeros, scales

def merge_qkv(model_config):
    """Merges the qkv fields in model_config from QKVConfig to a single LinearConfig."""
    for decoder_config in model_config.layers:
        if isinstance(decoder_config.attention.qkv, QKVConfig):
            splitted_qkv = decoder_config.attention.qkv
            decoder_config.attention.qkv = LinearConfig()
            decoder_config.attention.qkv.weight = splitted_qkv.weight
            decoder_config.attention.qkv.bias = splitted_qkv.bias
            decoder_config.attention.qkv.activation_scaling_factor = (
                splitted_qkv.activation_scaling_factor
            )
            decoder_config.attention.qkv.weights_scaling_factor = (
                splitted_qkv.weights_scaling_factor
            )
            decoder_config.attention.qkv.weights_scaling_factor_2 = (
                splitted_qkv.weights_scaling_factor_2
            )
            decoder_config.attention.qkv.prequant_scaling_factor = (
                splitted_qkv.prequant_scaling_factor
            )
            decoder_config.attention.qkv.awq_block_size = splitted_qkv.awq_block_size


            
def to_quantized_weight(
    weight: torch.Tensor, weights_scaling_factor: torch.Tensor, quantization: str
):
    
    print("---to_quantized_weight-----")
    """Converts the weight to the quantized (packed) format."""
    # Convert the tensor to CPU to avoid potential GPU OOM.
    weight = weight.cpu()
    weights_scaling_factor = weights_scaling_factor.cpu()


    if quantization == QUANTIZATION_FP8:
        # safe tensors does not support fp8 yet. So we pack the tensors as int8
        return (weight / weights_scaling_factor).to(torch.float8_e4m3fn).view(torch.int8)
    
    print(weight.shape)
    print( weights_scaling_factor.shape)
    
    if quantization == QUANTIZATION_INT8_Mix:
        return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)
    
    if quantization == QUANTIZATION_INT8_SQ:
        return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)

    if quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
        out_dim = weight.shape[0]
        assert (
            out_dim % 2 == 0
        ), f"Cannot pack weight. Out dimension {out_dim} is not an even number."
        in_dim = weight.shape[1]
        block_size = weight.shape[1] // weights_scaling_factor.shape[1]
        int8_tensor = (
            (weight / weights_scaling_factor[:, torch.arange(in_dim) // block_size])
            .round()
            .clamp(-8, 7)
            .to(torch.int8)
        )

        int8_tensor = int8_tensor.T.reshape(in_dim, out_dim // 2, 2)
        int4x2_tensor = (int8_tensor[:, :, 0] & 0x0F) | (int8_tensor[:, :, 1] << 4)
        # The shape of the returned weight is [out_dim // 2, in_dim]
        return int4x2_tensor.T.contiguous()

    raise NotImplementedError(f"quantization format {quantization} not supported")


def from_quantized_weight(
    weight: torch.Tensor, weights_scaling_factor: torch.Tensor, quantization: str, torch_dtype
):
    """Converts the quantized weight to the target torch_dtype format."""
    if weight.element_size() >= 2 or weights_scaling_factor is None or not quantization:
        # No need to unquantize the weight.
        return weight.to(torch_dtype)

    if quantization == QUANTIZATION_FP8:
        # safe tensors does not support fp8 yet. So we pack the tensors as int8
        return weight.view(torch.float8_e4m3fn).to(torch_dtype) * weights_scaling_factor.to(
            torch_dtype
        )

    if quantization == QUANTIZATION_INT8_SQ:
        return weight.to(torch_dtype) * weights_scaling_factor[:, None].to(torch_dtype)

    raise NotImplementedError(f"quantization format {quantization} not supported")

def pseudo_quantize_tensor(self, w: torch.Tensor, get_scale_zp=False):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2

        # zero point quantization
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** self.w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
        assert torch.isnan(w).sum() == 0

        w = w.reshape(org_w_shape)

        if get_scale_zp:
            return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
        else:
            return w
def pack_linear_weights(model_config: ModelConfig):
    """Packs the quantized linear weights in the model_config to the quantized format."""
    if not model_config.quantization:
        return
    print("pack_linear_weights")
     
    #print(model_config)

    #relative_path = "act_scales/%s.pt"%(model_config._name_or_path.split("/")[-1])

    relative_path = "act_scales/Llama-2-7b.pt"
    act_scales = torch.load(relative_path)
    from safetensors import safe_open
    awq_model = torch.load("/code/tensorrt_llm/manual_plugin/checkpoint/Llama-2-7b-w4-g128-v2.pt",
                          map_location=torch.device('cpu'))
      
    layer_id = -1
    #print(len(model_config.layers))
    for decoder_config in model_config.layers:
        linear_layers = [
            decoder_config.attention.qkv ,
            # decoder_config.attention.dense,
            # decoder_config.mlp.fc,
            # decoder_config.mlp.gate,
            # decoder_config.mlp.proj,
        ]

        names = [ "self_attn.q_proj",
                 
                 ]
        
        
        layer_id += 1
        print("layer id is ", layer_id)
        for i, linear_layer in enumerate(linear_layers):
            #print(model_config.quantization)
            
            

            if isinstance(linear_layer, LinearConfig):
                
                
                name = names[i]
                
                layer_scales = act_scales['model.layers.{}.{}'.format(layer_id, name)]

                # if "q_proj" in name:

                #     def get_qvk(layer_id,   name):
                #         awq_name = "model.layers.%d.self_attn.q_proj.%s"%(layer_id,  name)
                #         q1 = awq_mode[awq_name]
                #         awq_name = "model.layers.%d.self_attn.k_proj.%s"%(layer_id,   name)
                #         q2 = awq_model.get_tensor(awq_name)
                #         awq_name = "model.layers.%d.self_attn.v_proj.%s"%(layer_id,  name)
                #         q3 = awq_model.get_tensor(awq_name)
                #         return [q1, q2, q3]

                #     qweight = torch.cat(get_qvk(layer_id, "qweight"), dim=1)
                #     print(qweight.shape)
                #     qzeros = torch.cat(get_qvk(layer_id, "scaled_zeros"), dim=1)
                #     print(qzeros.shape)
                #     scales = torch.cat(get_qvk(layer_id, "scales") , dim=1)
                #     print(scales.shape)
                    
                #     import awq_ext
                #     fpweight = awq_ext.dequantize_weights_cuda(
                #     qweight.cuda().to(torch.int32), scales.cuda().to(torch.float16), 
                #     qzeros.cuda().to(torch.int32), 0, 0, 0, False
                #     )
                #     print("fpweight is ")
                #     print(fpweight)
                #     print("grand is")
                #     print(linear_layer.weight)
                #     y = np.loadtxt("/code/tensorrt_llm/manual_plugin/Input.csv")
                #     x = torch.as_tensor(y).to(torch.float16).to('cuda')
                #     x = torch.rand_like(x)/10
                #     x = x[0,:]
                #     print(x.shape)
                #     print("input x")
                #     awq =   torch.matmul(x, fpweight)
                #     print("out is")
                #     print(awq)
                #     print(" grand is ")
                #     print(linear_layer.weight.shape)
                #     grad =   torch.matmul(x, linear_layer.weight.T)
                #     print(grad)
                #     print("error is ")
                #     print(grad - awq)
                #     exit(0)

                # else:
                #     raise NotImplementedError("!")
                print("layer_scales")
                 
                linear_layer.weights_scaling_factor =   (torch.max(torch.abs(linear_layer.weight), dim=1)[0].unsqueeze(1) / (
                        127)).to(torch.float16).reshape((linear_layer.weight.shape[0],))
                

                ## to be done!
                ## pack the outliers to zeros so that we do not need to set the zeros!

                if linear_layer.weights_scaling_factor is not None:
                    # print("un quant weight ")
                    # print(linear_layer.weight)
                    # linear_layer.weight.data, scales, zeros = pseudo_quantize_tensor(
                    #     linear_layer.weight.data, 
                    #     get_scale_zp=True
                    # )
                    # scales = scales.t().contiguous()
                    # zeros = zeros.t().contiguous()
                
                    # qweight, qzeros, scales = awq_from_linear(linear_layer,  
                    #                                           qweight, 
                    #                                           qzeros, 
                    #                                           scales)
                    
                    import mixlib
                    # 一个int 放在2个half 中

                    linear_layer.qweight = mixlib.int_matrix_to_half(qweight.cuda()).cpu()
                    linear_layer.qzeros = mixlib.int_matrix_to_half(qzeros.cuda()).cpu() 
                    linear_layer.scales =   scales 


                    fp_features = 128

                    linear_layer.fp_ind =  torch.sort(layer_scales)[1][-fp_features:] 

                    print([layer_scales[linear_layer.fp_ind]])
                 
                    linear_layer.fp_weight = linear_layer.weight[:, linear_layer.fp_ind]
                    #linear_layer.weight[:, linear_layer.fp_ind] *= 0 # setting to zeros
                    # 转成short
                    import mixlib
                    # 一个int 放在2个half 中
                    linear_layer.fp_ind = mixlib.int_to_half(linear_layer.fp_ind.to(torch.int32).cuda()).cpu()
                    

                    linear_layer.weight = to_quantized_weight(
                        linear_layer.weight,
                        linear_layer.weights_scaling_factor,
                        model_config.quantization,
                    )
                    # print("q weight is ")
                    # print(linear_layer.weight)
                    # print("recorver weight is ")
                    # tmp = linear_layer.weight.to(torch.float16) * linear_layer.weights_scaling_factor[:, None].cpu()
                    # print(tmp[0,0:10])
                    # exit
 
