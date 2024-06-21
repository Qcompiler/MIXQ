

if [ $2 == a100 ]
    then
    CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
    else
    CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
fi
set -ex
quantpath=/home/dataset/quant/quant
modelpath=/home/dataset
 

models=(  "Llama-2-7b"  ) 
ngpu=1


data_type=$3
if [ ${data_type} == mix8  ]
    then 
    bit=${data_type:3:3}
    for model in "${models[@]}"
    do
        echo ${model}      

        CUDA_VISIBLE_DEVICES=$1    ${CMD} mmlu.py  \
        --model_type ${data_type}  --hf_model_dir  ${quantpath}${bit}/${model}

    done
fi

if [ ${data_type} == fp16  ]
    then 

    for model in "${models[@]}"
    do
        echo ${model}      

        CUDA_VISIBLE_DEVICES=$1    ${CMD} mmlu.py  \
        --model_type ${data_type}  --hf_model_dir  ${modelpath}/${model}

    done
fi

