
set -e
for model in Llama-2-7b  Llama-2-13b  Aquila2-7b falcon-7b Baichuan2-7b Baichuan2-13b
    do
    for batch_size in  8 16 32 64 128 256 512
    do

        CUDA_VISIBLE_DEVICES=$1 http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  python   benchflops.py --model_type mix \
        --model_path /data/chenyidong/checkpoint/quant/${model} --quant_file /data/chenyidong/checkpoint/quant/${model} \
        --batch_size ${batch_size}  
        
        CUDA_VISIBLE_DEVICES=$1 http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  python   benchflops.py --model_type awq \
        --model_path /data/chenyidong/checkpoint/awqquant/${model} --quant_file /data/chenyidong/checkpoint/awqquant/${model} \
        --batch_size ${batch_size}  

       CUDA_VISIBLE_DEVICES=$1 http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  python   benchflops.py --model_type bitsandbytes \
        --model_path /data/chenyidong/checkpoint/${model} --quant_file /data/chenyidong/checkpoint/${model} \
        --batch_size ${batch_size}  

        CUDA_VISIBLE_DEVICES=$1 http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  python   benchflops.py --model_type fp16 \
        --model_path /data/chenyidong/checkpoint/${model} --quant_file /data/chenyidong/checkpoint/${model} \
        --batch_size ${batch_size}     






    done
 done