
import os
import sys

import awq
os.environ["WORLD_SIZE"] = "1"
import time
import torch
import argparse
import numpy as np
import pandas as pd
from awq import AutoAWQForCausalLM
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

def run_round(model_path, quant_file, n_generate, token, batch_size, safetensors, model_type='fp16',cache=None):
    if model_type == 'mix':
        from mixquant import AutoForCausalLM
        model = AutoForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            max_new_tokens=n_generate, batch_size=batch_size,
            safetensors=safetensors,
            mix = True,
            cache = cache
        )



    if model_type == 'awq':
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
        
        model = model.to('cuda')

    if model_type == 'bitsandbytes':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        trust_remote_code=True,
        max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')

    print(f" -- Warming up...")
    warmup(model)

    print(f" -- Generating {n_generate} tokens,  in context...")

    try:
        generate_time = generate(model, token, n_generate, batch_size, cache)
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
        "Memory (VRAM)": f"{memory_used:.2f} GB ({memory_pct:.2f}%)"
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
    
    cache = MixLibCache()

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
    df.to_csv('output/throughput/'+args.model_type + '/' + args.quant_file.split("/")[-1]+".csv"+str(args.batch_size))

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
    args = parser.parse_args()

    main(args)