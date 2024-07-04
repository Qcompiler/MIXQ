# MixQ

MIXQ: Taming Dynamic Outliers in Mixed-Precision Quantization by Online Prediction


## Benchmarking the throughput



It is very easy to quantize a LLM and run by MIXQ 4bit or 8bit kernel

Running the following CMD to quantize the LLM with W8A8O16 kernel: 

```
python examples/basic_quant_mix.py --model_path /mnt/data/checkpoint/Llama-2-7b --quant_file /home/dataset/quant/quant8/Llama-2-7b --w_bit 8
```

Benchmark the throughput of MIXQ by:

```
python benchflops.py --model_type mix --model_path /home/dataset/quant/quant8/Llama-2-7b --quant_file /home/dataset/quant/quant8/Llama-2-7b --batch_size 512 --bit 8 
```

In  NVIDIA A100-PCIE-40GB, the output is

```
Version: mix 8bit 
|   Batch Size |   Decode Length |   Decode tokens/s | Memory (VRAM)    |
|-------------:|----------------:|------------------:|:-----------------|
|          512 |            1024 |           10609.8 | 7.86 GB (19.97%) |
```



# News !!

We have integrate the MixedQLinear  designed by QUIK into our framework! The QUIK now is able to support a wide range of LLMs including:


- Llama-2 7B/13B/70B
- Llama-3 8B
- Falcon 7B/40B
- ChatGLM 7B
- QWen2 7B


## How to Run

It is very easy to quantize a LLM and run by QUIK 4bit kernel

Running the following CMD to quantize the LLM 

```
python examples/basic_quant_quik.py --model_path /mnt/data/checkpoint/Llama-2-7b --quant_file /home/dataset/quant/quantquik4/Llama-2-7b --w_bit 4
```

Benchmark the throughput of QUIK by:

```
python  benchflops.py  --model_type quik --model_path   /home/dataset/quant/quantquik4/Llama-2-7b \
             --quant_file /home/dataset/quant/quantquik4/quik4/Llama-2-7b \
             --batch_size 512 --bit 4
```

In  NVIDIA A100-PCIE-40GB, the output is

```
Version: quik 4bit
|   Batch Size |   Decode Length |   Decode tokens/s | Memory (VRAM)    |
|-------------:|----------------:|------------------:|:-----------------|
|          512 |            1024 |           8981.17 | 4.88 GB (12.40%) |
```



# Tensorrt-LLM implementation

