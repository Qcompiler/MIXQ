
import os

os.environ["WORLD_SIZE"] = "1"
import sys
sys.path.append('/home/chenyidong/quant/AutoAWQ')
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# model_path = '/mnt/data/zhongrx/Llama-2-13b-hf'
# quant_path = '/mnt/data/chenyd/Llama-2-13b-awq'


model_path = '/mnt/data/zhongrx/Llama-2-7b-hf'
quant_path = '/data/chenyidong/Llama-2-7b-hf-mix'

quant_config = { "w_bit": 8, "version": "MIX" }
print(quant_path)
# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(model_path, mix = True, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize_mix(tokenizer, quant_config=quant_config)

# Save quantized model
# NOTE: pass safetensors=True to save quantized model weights as safetensors
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Mix Model is quantized and saved at "{quant_path}"')