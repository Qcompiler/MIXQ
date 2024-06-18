
import os
import sys


os.environ["WORLD_SIZE"] = "1"
import time
import torch
import argparse
import numpy as np
import pandas as pd


from transformers import AutoTokenizer
from torch.cuda import OutOfMemoryError
torch.manual_seed(0)

from mixquant.Cache import MixLibCache
def warmup(model):
    warm_up = torch.randn((4096,4096)).to(next(model.parameters()).device)
    torch.mm(warm_up,warm_up)






def prepare_data(_dataset_path = 'wikitext', _split='test', _text_column='text'):
    from datasets import load_dataset
    """
    Prepares the dataset by loading and formatting.

    Returns
    -------
    str
        The formatted dataset as a single string.
    """
    if _dataset_path == 'wikitext':
        _dataset_name = 'wikitext-2-raw-v1'
    if _dataset_path == 'c4':
        _dataset_name = 'realnewslike'        
    # Load the dataset
    data = load_dataset(_dataset_path, _dataset_name, split=_split)
    # Format the text column of the dataset
    text_list = [' \n' if s == '' else s for s in data[_text_column]]
    return ''.join(text_list)
    
def decode_token(model, _tokenizer, _text, n_batch, repeat = 10):


    tokens = _tokenizer(_text, truncation=False, return_tensors='pt').input_ids.to('cuda')
    start = 0
    end = n_batch
    for j in range(repeat):

        batch_start = start + j * n_batch
        batch_size = min(end - batch_start, n_batch)

        token_org = tokens[0][batch_start].item()

        if j == 0:
            # Replace the first token with the BOS token
            tokens[0][batch_start] = _tokenizer.bos_token_id

        # Compute the logits for the current batch of tokens
        _compute_batch_logits(tokens, batch_start, batch_size)

        tokens[0][batch_start] = token_org

def _compute_batch_logits(_model,tokens, batch_start, batch_size):
    # Compute the logits without keeping track of gradients

    outputs = _model(tokens[:, batch_start:batch_start+batch_size])  
    return outputs


def generate(model, tokens, n_generate, batch_size, cache):
    context_time = 0
    generate_time = []
    

    with torch.inference_mode():


        # prefill context
        cache.is_prefill = False
        
        


        for i in range(10):
            batch_start = i * batch_size
            inputs = torch.as_tensor(tokens[:, batch_start:batch_start+batch_size], device=next(model.parameters()).device)
            inputs = inputs.reshape((batch_size,1,))
            out = model(inputs,use_cache=True)

            



    with torch.inference_mode():
        # cache.is_prefill = True
        # inputs = torch.as_tensor(input_ids, device=next(model.parameters()).device)
        # out = model(inputs,use_cache=True)
        # token = out[0][:, -1].max(1)[1].unsqueeze(1)

        for i in range(n_generate):
            batch_start = i * batch_size
            torch.cuda.synchronize()
            

            # decode tokens
            cache.is_prefill = False
            inputs = torch.as_tensor(tokens[:, batch_start:batch_start+batch_size], device=next(model.parameters()).device)
            inputs = inputs.reshape((batch_size,1,))
            start = time.time()
            

            out = model(inputs,use_cache=True)
            torch.cuda.synchronize()            


            generate_time.append(time.time() - start)


    print("--- generate time ---")
    #print(generate_time)
    return  generate_time

def run_round(model_path, quant_file, n_generate, token, batch_size, safetensors, model_type='fp16',mixlibcache=None):
    if model_type == 'mix':
        from mixquant import AutoForCausalLM
        model = AutoForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            max_new_tokens=n_generate, batch_size=batch_size,
            safetensors=safetensors,
            mix = True,
            cache = mixlibcache
        )



    if model_type == 'awq':

        import awq
        from awq import AutoAWQForCausalLM
        print(f" -- Loading model awq...")
        model = AutoAWQForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            max_new_tokens=n_generate, batch_size=batch_size,
            safetensors=safetensors
        )
    if model_type == 'fp16':    
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map='auto', trust_remote_code=True
        )
        


    if model_type == 'bitsandbytes':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        trust_remote_code=True,
        max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')



    if model_type == 'quik':
        args.fp_features_num = 256
        sys.path.append("/home/chenyidong/quant/QUIK/experiments")
        def get_fp_features_num(module: torch.nn.Linear, args):
            fp_features_num = args.fp_features_num
            return fp_features_num
        def llama_replace_with_kernels(model, args):
            import modelutils
            import quant_sim
            import qlinear
            layers = model.model.layers
            shared_inputs = {}

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



        import modelutils, quant_sim  
        model = modelutils.get_llama(args.model_path, args.batch_size, "")
        print("Load quantized model from ", args.quant_file)
        save_dict = torch.load(args.quant_file)
        model.load_state_dict(save_dict["model"])   
        model.config.use_cache = True
        model = model.to('cuda')
        cache = {'past': None}
        def clear_past(i):
            def tmp(layer, inp, out):
                if cache['past']:
                    cache['past'][i] = None
            return tmp
        for i, layer in enumerate(model.model.layers):
            layer.register_forward_hook(clear_past(i))        

        
        relative_path = "/home/chenyidong/quant/QUIK/experiments/act_scales/{}.pt".format(args.model_path.split('/')[-1])

        print(relative_path)
        act_scales = torch.load(relative_path)


 
        quant_sim.add_actquant(model)
        layers = modelutils.find_layers(model)

        for name in layers:
            
            bits = 4
            if 'lm_head' in name or "rotary_emb" in name:
                print(f'Skipping {name}\n')
                continue 
            
            
            if 'down_proj' in name:
                bits = 8       
            
            if args.fp_features_num > 0 :
                fp_features_num = get_fp_features_num(layers[name].module, args)
                if "qkv" in name:
                    act_name = name.replace("qkv", "q")
                else:
                    act_name = name
                layers[name].fp_features_configure(act_scales[act_name], fp_features_num)
            layers[name].quantizer.configure(bits=bits)

        llama_replace_with_kernels(model, args)    
        model = model.to('cuda')


    print(model)
    print(f" -- Warming up...")
    warmup(model)

    print(f" -- Generating {n_generate} tokens,  in context...")

    try:
        generate_time = generate(model, token, n_generate, batch_size, mixlibcache)
        successful_generate = True
    except RuntimeError as ex:
        if 'cuda out of memory' in str(ex).lower():
            successful_generate = False
        else:
            raise RuntimeError(ex)
    
    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    if successful_generate:
        # number of tokens in context / time for processing context * batch size
        # 1 second / median time per token in seconds * batch size
        decode_tokens_per_second = 1 / np.median(generate_time) * batch_size

        print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
        print(f" ** Max Memory (VRAM): {memory_used:.2f} GB ({memory_pct:.2f}%)")
    else:

        decode_tokens_per_second = 'OOM'

    return {
        "Batch Size": batch_size,
        "Decode Length": n_generate,
        "Decode tokens/s": decode_tokens_per_second,
        "Memory (VRAM)": f"{memory_used:.2f} GB ({memory_pct:.2f}%)",
        "latency" : generate_time
    }, args.model_type

def main(args):
    rounds = [
        # {"context": 32, "n_generate": 32},
        # {"context": 64, "n_generate": 64},
        # {"context": 128, "n_generate": 128},
        # {"context": 256, "n_generate": 256},
        # {"context": 512, "n_generate": 512},
        {"context": args.seq_length, "n_generate": args.seq_length},
        # {"context": 2048, "n_generate": 2048},
    ]

    all_stats = []
    
    cache = MixLibCache(bit=args.bit)

    print("downloading data......")
    text = prepare_data()
    print("done......")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id    
    tokenizer.model_max_length = sys.maxsize
    tokens = tokenizer(text, truncation=False, return_tensors='pt').input_ids.to('cuda')

    

    

    
    for settings in rounds:
         


        stats, model_version = run_round(
            args.model_path,
            args.quant_file,
            settings["n_generate"],
            tokens,
            args.batch_size,
            args.safetensors,
            args.model_type,
            cache
        )
        
        all_stats.append(stats)

        if stats["Decode tokens/s"] == 'OOM':
            break
    
    df = pd.DataFrame(all_stats)
    print('GPU:', torch.cuda.get_device_name())
    print('Model:', args.model_path)
    print('Version:', model_version)
    print(df.to_markdown(index=False))
    try:
        os.mkdir('output/throughput/'+args.model_type)
    except:
        pass
    df.to_csv('output/throughput/'+args.model_type + '/' + args.quant_file.split("/")[-1] \
              + str(args.batch_size) + '_' +  str(args.bit) + ".csv")

if __name__ == "__main__":

    """
    python examples/benchmark.py --model_path /mnt/data/zhongrx/Llama-2-7b-hf --quant_file /mnt/data/chenyd/Llama-2-7b-awq 

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="casperhansen/vicuna-7b-v1.5-awq", help="path to the model")
    parser.add_argument("--quant_file", type=str, default="awq_model_w4_g128.pt", help="weights filename")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for cache and generation")
    parser.add_argument("--model_type", type=str, default="awq")
    parser.add_argument("--safetensors", default=False, action="store_true", help="Use for enabling safetensors")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--seq_length", type=int, default=32)
    parser.add_argument("--bit", type=int, default=8)
    args = parser.parse_args()

    main(args)