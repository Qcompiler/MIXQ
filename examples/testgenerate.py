


import os
import time
import psutil
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter

from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelForCausalLM

 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def test_from_fp16():
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)

    # model_name = '/root/data/models/2023/llama-13B-v1/'
    model_name = '/mnt/octave/data/chenyidong/checkpoint/Llama-2-7b'
    MAX_NEW_TOKENS = 32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    # config.num_hidden_layers = 1

    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-1}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
    model.eval()

    # from eetq.utils import eet_accelerator
    # eet_accelerator(model, quantize=True, fused_attn=True, dev="cuda:0")
    model.to("cuda:0")
    # for k, v in model.state_dict().items():
    #     print(k, v.shape, v.dtype, v.device)
    # torch.save(model, "eetq_llama13B_model_fused_attn_v2.pt")

    prompt_template = "[INST] {prompt} [/INST]"

    prompt = "You're standing on the surface of the Earth. "\
            "You walk one mile south, one mile west and one mile north. "\
            "You end up exactly where you started. Where are you?"

 
    messages = []
    messages.append({"role": "user", "content": prompt})
    response = model.chat(tokenizer, messages)
    print(response)
test_from_fp16()