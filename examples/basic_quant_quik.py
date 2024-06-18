
import os

os.environ["WORLD_SIZE"] = "1"
import sys
from mixquant import AutoForCausalLM
from transformers import AutoTokenizer

import argparse
parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
parser.add_argument("--model_path", type=str,   help="Model path")
parser.add_argument("--quant_file", type=str,   help="quant_file Model path")
parser.add_argument("--w_bit", type=int, default=4,  help="weight bit")
args = parser.parse_args()

model_path = args.model_path
quant_path = args.quant_file
quant_config = { "w_bit": args.w_bit, "version": "QUIK" }
print(quant_path)
# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoForCausalLM.from_pretrained(model_path, mix = True, **{"low_cpu_mem_usage": True},device_map='cpu')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(model)
# Quantize
model.quantize_quik(tokenizer, quant_config=quant_config)

# Save quantized model
# NOTE: pass safetensors=True to save quantized model weights as safetensors
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'QUIK Model is quantized and saved at "{quant_path}"')