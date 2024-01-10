
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WORLD_SIZE"] = "1"
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# model_path = '/mnt/data/zhongrx/Llama-2-13b-hf'
# quant_path = '/mnt/data/chenyd/Llama-2-13b-awq'


model_path = '/mnt/data/huangkz/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546'
quant_path = '/mnt/data/chenyd/models--facebook--opt-30b-awq'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
print(quant_path)
# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
# NOTE: pass safetensors=True to save quantized model weights as safetensors
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')