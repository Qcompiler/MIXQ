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

from mixquant.utils.module import get_named_linears, set_op_by_name, weight_only_map


class MixQuantizer:
    def __init__(self, f16_model, model, tokenizer, w_bit, group_size, version) -> None:
        self.f16_model = f16_model
        self.model = model
        self.tokenizer = tokenizer

        self.group_size = group_size
        self.version = version

        self.modules, self.module_kwargs= self.init_quant()
    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.f16_model.get_model_layers(self.model)


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

        modules[0] = Catcher(modules[0])

        modules[0] = modules[0].module  # restore

        modules[0] = modules[0].cpu()
        self.f16_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if "attention_mask" in layer_kwargs.keys():
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs
    

    def quantize(self,weight_only = False):
        for i in tqdm(range(len(self.modules)), desc="Mix quant"):
            # [STEP 1]: Get layer, extract linear modules, extract input features
            self.modules[i] = self.modules[i].cuda()
            named_linears = get_named_linears(self.modules[i])

            clear_memory()

            # Quantize weights
            self._apply_quant(self.modules[i], named_linears, weight_only)
            clear_memory()



    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear], weight_only_):

        
        if isinstance(self.model.config.architectures,list):
            name = self.model.config.architectures[0]
        else:
            name = self.model.config.architectures
        weight_only_name = weight_only_map[ name ]

        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            if self.version == 'MIX':
                q_linear_module = MixLinear_GEMM

            else:
                raise NotImplementedError
            
            # for same small blocks we do not need the mixquant, we only use the weight only quant

            weight_only = False

            for key in weight_only_name:
                if key in  name:
                    weight_only = True
                    break


            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                weight_only = weight_only,
                init_only=False
            )

            linear_layer.cpu()

            set_op_by_name(module, name, q_linear)
            clear_memory()