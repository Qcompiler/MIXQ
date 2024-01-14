from .base import BaseForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM, GPTJBlock

class GPTJMixForCausalLM(BaseForCausalLM):
    layer_type = "GPTJBlock"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model: GPTJForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def get_act_for_scaling(module: GPTJBlock):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.fc_in.out_features
        )
    
    @staticmethod
    def move_embed(model: GPTJForCausalLM, device: str):
        model.transformer.wte = model.transformer.wte.to(device)
    
