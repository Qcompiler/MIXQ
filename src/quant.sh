

CMD="srun -p twills -A h100 --gres=gpu:h100:1"
 
 
set -x

model=65      
CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
python examples/basic_quant_mix.py  \
--model_path /home/dataset/llama-2/checkpoint/Llama-${model}b \
--quant_file /home/dataset/llama-2/checkpoint/quant/Llama-${model}b


# model=13      
# CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
# python examples/basic_quant_mix.py  \
# --model_path /home/dataset/llama-2/Llama-2-${model}b-hf \
# --quant_file /home/chenyidong/quant/Llama-2-${model}b-hf



# model=70      
# CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
# python examples/basic_quant_mix.py  \
# --model_path /home/dataset/llama-2/Llama-2-${model}b-hf \
# --quant_file  /home/chenyidong/quant/Llama-2-${model}b-hf

# for model in 7 34
# do  
#     CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
#     python examples/basic_quant_mix.py  \
#     --model_path /home/dataset/llama-2/checkpoint/Aquila2-${model}b \
#     --quant_file /home/dataset/llama-2/checkpoint/quant/Aquila2-${model}b
# done 

# for model in 6.7  13
# do  
#     CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
#     python examples/basic_quant_mix.py  \
#     --model_path /home/dataset/llama-2/checkpoint/opt-${model}b \
#     --quant_file /home/dataset/llama-2/checkpoint/quant/opt-${model}b
# done 


for model in 7
do  
    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
    python examples/basic_quant_mix.py  \
    --model_path /home/dataset/llama-2/checkpoint/Mistral-${model}b \
    --quant_file /home/dataset/llama-2/checkpoint/quant/Mistral-${model}b
done 


