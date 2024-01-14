import torch
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from mixquant.utils.utils import clear_memory
from mixquant.utils.calib_data import get_calib_dataset
from mixquant.modules.linear import MixLinear_GEMM
from mixquant.utils.module import get_named_linears, set_op_by_name



class MixQuantizer:
    def __init__(self, f16_model, model, tokenizer, w_bit, group_size, version, 
                       calib_data, split, text_column) -> None:
        self.f16_model = f16_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.modules, self.module_kwargs, self.inps = self.init_quant()
    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.f16_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data, tokenizer=self.tokenizer, n_samples=n_samples, block_size=seqlen,
            split=self.split, text_column=self.text_column
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        modules[0] = modules[0].cuda()
        self.f16_model.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hijacked_inputs, **kwargs):
                inps.append(hijacked_inputs)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])


        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        modules[0] = modules[0].module  # restore
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.f16_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if "attention_mask" in layer_kwargs.keys():
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs, inps
    

    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="Mix quant"):
            # [STEP 1]: Get layer, extract linear modules, extract input features
            self.modules[i] = self.modules[i].cuda()
            named_linears = get_named_linears(self.modules[i])

            clear_memory()

            # Quantize weights
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()

    def pseudo_quantize_tensor(self,weight):
        weight_scale = weight.abs().max() / 127       
        weigwht_scaled = (weight / weight_scale)
        return weigwht_scaled, weight_scale

    def get_scales_of_each_weight(self,weight):
        weight_scale = weight.abs().max() / (2 ** (self.w_bit - 1) - 1)       

        return weight_scale
    
 


    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        scales = {}
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            scales[name] = self.get_scales_of_each_weight(linear_layer.weight.data)
        
        
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            if self.version == 'MIX':
                q_linear_module = MixLinear_GEMM

            else:
                raise NotImplementedError
            
            weight_only = False
            if "o_proj" in name or "down_proj" in name:
                weight_only =  True


            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                weight_only = weight_only,
                init_only=False
            )

            linear_layer.cpu()

            set_op_by_name(module, name, q_linear)
            clear_memory()