export PYTHONPATH=$PYTHONPATH:/home/chenyidong/MIXQ

if [ $2 == a100 ]
    then
    CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
    #CMD="  python "
fi
if [ $2 == direct ]
    then
    CMD="  python "
    #CMD="  python "
fi

if [ $2 == h100 ]
    then
    CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
fi
if [ $2 == 4090 ]
    then
    CMD=" srun -N 1 --gres=gpu:4090:1 --pty  python"
fi
#CMD=" python" 
 
set -x

# model=65      
# CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
# python examples/basic_quant_mix.py  \
# --model_path /home/dataset/llama-2/checkpoint/Llama-${model}b \
# --quant_file /home/dataset/llama-2/checkpoint/quant/Llama-${model}b


models=(  "Baichuan2-7b"  "Baichuan2-13b" "Aquila2-7b" "Llama-2-7b"  "Mistral-7b" )
models=(  "llama-2-hf"    )
models=(  "falcon-40b"   )
models=(   "Llama-2-7b"  "falcon-7b" "vicuna-7b" "chatglm2-6b" )
quantpath=/home/chenyidong/data/mixqdata/quant
modelpath=/home/chenyidong/data/mixqdata

for bit in   8 
  do 
    for model in "${models[@]}"
            do
            echo ${model}
            ${CMD}   examples/basic_quant_mix.py  \
            --model_path ${modelpath}/${model} \
            --quant_file ${quantpath}${bit}/${model} --w_bit ${bit}
    done
done
 

# for bit in 4 
#   do 
#     for model in "${models[@]}"
#             do
#             echo ${model}
#             ${CMD} \
#               examples/basic_quant_quik.py  \
#             --model_path ${modelpath}/${model} \
#             --quant_file ${quantpath}quik${bit}/${model} --w_bit ${bit}
#     done
# done
 

