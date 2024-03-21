

CMD="srun -p twills -A h100 --gres=gpu:h100:1 python"
CMD=" python" 
 
set -x

# model=65      
# CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
# python examples/basic_quant_mix.py  \
# --model_path /home/dataset/llama-2/checkpoint/Llama-${model}b \
# --quant_file /home/dataset/llama-2/checkpoint/quant/Llama-${model}b


models=(  "Baichuan2-7b"  "Baichuan2-13b" "Aquila2-7b" "Llama-2-7b"  "Mistral-7b" )
models=(  "vicuna-33b"  "vicuna-7b" ) 
for model in "${models[@]}"
        do
        echo ${model}
        CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
          examples/basic_quant_mix.py  \
        --model_path /mnt/octave/data/chenyidong/checkpoint/${model} \
        --quant_file /mnt/octave/data/chenyidong/checkpoint/quant/${model}
done


