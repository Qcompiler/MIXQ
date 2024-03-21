import os
import sys

os.environ["WORLD_SIZE"] = "1"
import argparse
import pandas as pd
import torch
from utils.utils import Perplexity
from transformers import AutoTokenizer






def get_fp_features_num(module: torch.nn.Linear, args):
    fp_features_num = args.fp_features_num
    if args.fp_features_frac is not None:
        fp_features_num = max(int(module.in_features * args.fp_features_frac), fp_features_num)
    return fp_features_num
def llama_replace_with_kernels(model, args):
    import modelutils
    layers = model.model.layers
    shared_inputs = {}

    assert not args.w_asym, 'Benchmarking only supports symmetric weight quantization!'
    print("Replace with INT4 kernels.")
    for i in range(len(layers)):
        opt_block = layers[i]
        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]
        full = modelutils.find_layers(opt_block)
        for j, layer_group in enumerate(sequential):
            subset = {n: full[n] for n in layer_group}
            shared_inputs[f"{i}.{j}"] = qlinear.SharedQuantizedInput(len(layer_group))
            for name in subset:
                layer = subset[name]
                if 'lm_head' in name or 'rotary_emb' in name:
                    continue
                is_quantized = False
                bits = 16
                fp_features = 0
                import quant_sim
                import qlinear
                if isinstance(layer, quant_sim.ActQuantWrapper):
                    if layer.quantizer.configured:
                        is_quantized = True
                        bits = layer.quantizer.bits
                        fp_features = layer.fp_features_num
                    layer = layer.module
                layer_weight = layer.weight.data

                layer_scale = save_dict['model.layers.{}.{}.scale'.format(i, name)]
                if fp_features == 0:
                    fp_feature_idx = None
                else:
                    print('---------------save  act_scales----------------')
                    layer_act_scales = act_scales['model.layers.{}.{}'.format(i, name)]
                    fp_feature_idx = torch.sort(layer_act_scales)[1][-fp_features:]

                if is_quantized:
                    int_mod = qlinear.MixedQLinear.from_float(layer, layer_weight, layer_scale,
                                                              shared_inputs[f"{i}.{j}"], fp_feature_idx,
                                                              bits=bits)
                else:
                    int_mod = layer
                modelutils.replace_single_mod_opt(opt_block, name, int_mod)



from transformers import AutoTokenizer, TextStreamer
def promot(model, tokenizer):
    text = 'Hamburg is in which country?\n'
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}


    generated_ids = model.generate(input_ids, max_length=512)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python examples/benchmark/perplexity.py \
        --model_name  /mnt/data/zhongrx/Llama-2-7b \
        --model_basename gptq_model-4bit-128g \
        --is_quantized
    
    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_path", type=str,   help="Model path")
    parser.add_argument("--quant_file", type=str,   help="quant_file Model path")
    
    parser.add_argument("--model_type", type=str,  default='bitsandbytesfp16')


    parser.add_argument("--n_ctx", type=int, default=256, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default='wikitext', help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default='test', help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default='text', help="Column in the dataset containing the text.")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="Max memory used in each GPU.")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    
    parser.add_argument("--use_safetensors", action="store_true", help="Whether to use safetensors model file")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument("--disable_exllama", action="store_true", help="Whether to use disable exllama kernel")


    args = parser.parse_args()
    

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

 
    model_path = args.model_path
    quant_file = args.quant_file

    if args.model_type == 'bitsandbytesfp16':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model  fp16...")
        # model = transformers.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,  
        #                                                      device_map='auto')
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True
        )
        
        model = model.to('cuda')





    if args.model_type == 'mix':
        from mixquant.Cache import MixLibCache
        from mixquant import AutoForCausalLM
        cache = MixLibCache(args.n_batch)


        model = AutoForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            mix = True,  cache = cache
        )  

    if args.model_type == 'awq':
        print(f" -- Loading model awq...")
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            max_new_tokens=n_generate, batch_size=batch_size,
            safetensors=safetensors
        )
    if args.model_type == 'fp16':    
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map='auto', trust_remote_code=True
        )
        
        model = model.to('cuda')

    text = 'Hamburg is in which country?\n'
    tokens = tokenizer.encode_plus(text)['input_ids']
    tokens = torch.tensor(tokens)[None,].to('cuda')
    stop_tokens = ["###", "[UNK]", "</s>"]
    with torch.no_grad():
        out = model.generate(tokens, do_sample=True, max_length=512, 
                             bad_words_ids=[[tokenizer.encode(token)[0] for token in stop_tokens]])[0]
        out = tokenizer.decode(out.cpu().numpy().tolist())
        print(out)



