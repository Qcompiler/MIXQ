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

    parser.add_argument('--a_bits', type=int, default=4, choices=[4, 8, 16])

    # Weight Quantization Params: 
    parser.add_argument('--w_bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument('--w_clip', action='store_true', help='Use clipping for weight quantization')
    parser.add_argument('--w_asym', action='store_true')
    
    parser.add_argument('--int8_down_proj', action='store_true', help='Use INT8 for Down Projection')
    parser.add_argument('--fp_features_frac', type=float, default=None, help='Fraction of features to keep in FP16.')    
    parser.add_argument("--fp_features_num", type=int, default=1, help="outliers")

    parser.add_argument('--eval_accuracy', type=bool, default=True)
    parser.add_argument('--eval_throughput', type=bool, default=False)


    args = parser.parse_args()
    
    if args.eval_throughput is True:
        args.eval_accuracy = False

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
        print(model)

    if args.model_type == 'bitsandbytesmix4':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model  mix4...")
    
        n_gpus = torch.cuda.device_count()
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int4_threshold=6.0,
        llm_int4_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=quantization_config
        )
    if args.model_type == 'bitsandbytesmix8':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model mix bit8...")
    
        n_gpus = torch.cuda.device_count()
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=quantization_config
        )
    if args.model_type == 'awq':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model  awq...")
        sys.path.append("/home/chenyidong/quant/AutoAWQ")

        from awq import AutoAWQForCausalLM        
        model = AutoAWQForCausalLM.from_quantized(model_path, quant_file, fuse_layers=True,mix = False)


    if args.model_type == 'mix':
        from mixquant.Cache import MixLibCache
        from mixquant import AutoForCausalLM
        cache = MixLibCache(args.n_batch)


        model = AutoForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            mix = True,  cache = cache
        )         



    if args.model_type == 'QUIK':
        import modelutils    
        model = modelutils.get_llama(args.model_path, args.n_ctx, "")
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
            
            bits = args.a_bits
            if 'lm_head' in name or "rotary_emb" in name:
                print(f'Skipping {name}\n')
                continue 
            
            
            if 'down_proj' in name:
                if args.int8_down_proj:
                    bits = 8       
            
            if args.fp_features_num > 0 or args.fp_features_frac is not None:
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
    ppl = Perplexity(model, tokenizer, args.dataset_path, args.dataset_name, args.split, args.text_column, args.eval_accuracy)
    allppl = ppl.calculate_perplexity(args.n_ctx, args.n_batch)

    data = pd.DataFrame(allppl)
    try:
        os.mkdir("output")
    except:
        pass
    data.to_csv("output/ppl_batchsize"+str(args.n_ctx)+"_"+args.model_type+"_"+model_path.split('/')[-1]+".csv" + str(args.fp_features_num))

    


