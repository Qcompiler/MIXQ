

CMD="srun -N 1 --gres=gpu:4090:1  --pty python "
set -ex
basepath=/home/chenyidong/data/mixqdata
_dataset_path=/code/checkpoint/dataset


data_type=$1

 

models=(  "falcon-7b"  "vicuna-7b" "chatglm2-6b" ) 
ngpu=1
if [ ${data_type} == mix8 ]  
    then 
    for model in "${models[@]}"
    do

        echo ${model}      
        bit=${data_type:3:3}
        CUDA_VISIBLE_DEVICES=0    ${CMD} examples/mmlu.py  --model_type ${data_type} \
        --hf_model_dir  ${basepath}/quant8/${model}  \
        --data_dir  ${basepath}/data/data

    done
fi


if [ ${data_type} == fp16  ] || [ ${data_type} == bitsandbytes  ]
    then 
    for model in "${models[@]}"
    do
        echo ${model}      
        export TRANSFORMERS_VERBOSITY=error
        CUDA_VISIBLE_DEVICES=0    ${CMD} examples/mmlu.py  --model_type ${data_type} --hf_model_dir  ${basepath}/${model}  \
        --hf_model_dir  ${basepath}/${model}  \
        --data_dir  ${basepath}/data/data
    done
fi


if [ ${data_type} == awq  ]
    then 
 
    for model in "${models[@]}"
    do
        echo ${model}      

        CUDA_VISIBLE_DEVICES=0    ${CMD} examples/mmlu.py  --model_type ${data_type} \
        --hf_model_dir  ${basepath}/${model}-AWQ   --data_dir  ${basepath}/data/data

    done
    
fi

 
