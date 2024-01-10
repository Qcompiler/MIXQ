from .base import BaseForCausalLM
from typing import Dict
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

class LlamaMixQForCausalLM(BaseForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: LlamaForCausalLM, quant_config: Dict, mix = False, cache = None):
 
        fuser = LlamaFuser(model, quant_config)
        
        fuser.fuse_attention(MixGemmCache = cache)
        
        fuser.fuse_rmsnorm()
        fuser.fuse_mlp(mix, MixGemmCache = cache)

    @staticmethod
    def get_model_layers(model: LlamaForCausalLM):
        return model.model.layers
    
 
    
    @staticmethod
    def move_embed(model: LlamaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
  

import torch
from typing import List, Tuple, Union
from mixquant.utils.utils import set_module_name
from mixquant.modules.fused.mlp import  MixLlamaMLP
from mixquant.modules.fused.attn import QuantAttentionFused
from mixquant.modules.fused.norm import FasterTransformerRMSNorm
from mixquant.modules.linear import  MixLinear_GEMM

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP
import sys


#from  modeling_baichuan import Attention
class LlamaFuser:
    def __init__(self, model, quant_config):
        self.model = model
        self.quant_config = quant_config

        #print(model.model.layers[0].self_attn.o_proj) # 确认一下模型的权重的格式
 
        #需要加入百川的 Attention
        self.attention_modules: List[Tuple[str, LlamaAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaAttention) or  "Attention" in str(module.__class__)
        ]
        #print(self.attention_modules)

        self.rmsnorm_modules: List[Tuple[str, LlamaRMSNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaRMSNorm)   or  "RMSNorm" in str(module.__class__)
        ]
        
        self.mlp_modules: List[Tuple[str, LlamaMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaMLP)   or  "MLP" in str(module.__class__)
        ]
    
    def fuse_attention(self, MixGemmCache):
        for name, module in self.attention_modules:
            qkv_layer  = self._fuse_qkv(module)
            try:
                num_key_value_heads = module.num_key_value_heads
            except:
                # 为了处理百川的模型
                print("do not find the attr module.num_key_value_heads")
                num_key_value_heads = 32
            attn = QuantAttentionFused(
                module.hidden_size,
                module.num_heads,
                num_key_value_heads,
                qkv_layer, 
                module.o_proj,
                next(iter(qkv_layer.state_dict().values())).device,
                self.model.config.max_new_tokens,
                MixGemmCache = MixGemmCache
            )
            set_module_name(self.model, name, attn)
    
    def _fuse_qkv(self, module: LlamaAttention):
        try:
            q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
        except:
            qkv_layer = module.W_pack
            return qkv_layer
 
 
        
        if not  isinstance(q_proj, MixLinear_GEMM) :
            raise "no implement error"
 
        if isinstance(q_proj, MixLinear_GEMM):
            qkv_layer = MixLinear_GEMM(q_proj.in_features,q_proj.out_features + k_proj.out_features + v_proj.out_features,
                                        q_proj.bias is not None,next(iter(module.state_dict().values())).device)

 
 




 
        
        if isinstance(qkv_layer, MixLinear_GEMM):
            shapew = qkv_layer.weight.shape
            qkv_layer.weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)

            assert shapew[0] == qkv_layer.weight.shape[0]
            assert shapew[1] == qkv_layer.weight.shape[1]
            # shapew = qkv_layer.qweight.shape
            # qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=0)
            # qkv_layer.scale = q_proj.scale
            # assert shapew[0] == qkv_layer.qweight.shape[0]
            # assert shapew[1] == qkv_layer.qweight.shape[1]
            if q_proj.bias is not None:
                raise NotImplementedError
            else:
                qkv_layer.bias = None

        else:
            raise "no implement"
        

        return qkv_layer

    def fuse_rmsnorm(self):
        for name, module in self.rmsnorm_modules:
            norm = FasterTransformerRMSNorm(module.weight, module.variance_epsilon)
            set_module_name(self.model, name, norm)

    def fuse_mlp(self,mix, MixGemmCache = None):
        for name, module in self.mlp_modules:
            if  mix:
                assert MixGemmCache is not None
                mlp = MixLlamaMLP(module.gate_proj, module.down_proj, module.up_proj , MixGemmCache)
            set_module_name(self.model, name, mlp)