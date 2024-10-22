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
set -x

quantpath=/home/dataset/quant
modelpath=/home/dataset
dataset_path=/home/dataset/quant/checkpoint/dataset

model=$3
data_type=$4
down_weight_only=0

for batch in    512 
    do
    for seq in   64  
        do


            if [ ${data_type} == fp16 ]
                then 
                 
                   
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
                    ${modelpath}/${model} \
                    --quant_file ${modelpath}/${model} \
                    --n_ctx $batch --n_batch $batch  --eval_accuracy True --dataset_path ${dataset_path} 


                 
            fi

            if [ ${data_type} == bitsandbytes ]
                then 
                 
                   
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1  ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
                    ${modelpath}/${model} \
                    --quant_file ${modelpath}/${model} \
                    --n_ctx $batch --n_batch $batch  --eval_accuracy True --dataset_path ${dataset_path} 


                 
            fi

            if [ ${data_type} == awq ]
                then 
                    pip install transformers==4.35

                    echo ${model}
                    CUDA_VISIBLE_DEVICES=$1     \
                    ${CMD} evalppl.py --model_type ${data_type} --model_path  \
                    ${quantpath}/awqquant/${model} \
                    --quant_file ${quantpath}/awqquant/${model} \
                    --n_ctx $batch --n_batch $batch  --dataset_path ${dataset_path} --eval_accuracy True

                    pip install transformers==4.41.2
           
            fi

            if [ ${data_type} == mix8 ]
                then 
                    bit=8
                    echo  "---------run mix 8--------"
     
                        echo ${model}    
                    if [ ${down_weight_only} == 1 ]
                        then      
                        rm -r ${quantpath}/quant${bit}/down_weight_only/${model}/model.safetensors     
                        CUDA_VISIBLE_DEVICES=$1    \
                        ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                        ${quantpath}/quant${bit}/down_weight_only/${model} \
                        --quant_file ${quantpath}/quant${bit}/down_weight_only/${model} \
                        --n_ctx ${batch}  --n_batch $batch  --dataset_path ${dataset_path} --eval_accuracy True
                    fi
                    if [ ${down_weight_only} == 0 ]
                            then      
                        rm -r ${quantpath}/quant${bit}/${model}/model.safetensors     
                        CUDA_VISIBLE_DEVICES=$1    \
                        ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                        ${quantpath}/quant${bit}/${model} \
                        --quant_file ${quantpath}/quant${bit}/${model} \
                        --n_ctx ${batch}  --n_batch $batch  --dataset_path ${dataset_path} --eval_accuracy True
                    fi
            
            fi
            if [ ${data_type} == mix4 ]
                then 
                    bit=4
                    echo  "---------run mix 4--------"
                     
                        rm -r ${quantpath}/quant${bit}/down_weight_only/${model}/model.safetensors   
                        echo ${model}          
                        CUDA_VISIBLE_DEVICES=$1   \
                        ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                        ${quantpath}/quant${bit}/down_weight_only/${model} \
                        --quant_file ${quantpath}/quant${bit}/down_weight_only/${model} \
                        --n_ctx ${batch}  --n_batch $batch  --dataset_path ${dataset_path} --eval_accuracy True
   
                  
     
            fi
 
            if [ ${data_type} == quik ]
                then 
                    bit=4
                    echo  "---------run quik 4--------"
                     
                       
                        echo ${model}          
                        CUDA_VISIBLE_DEVICES=$1   \
                        ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                        ${quantpath}/quantquik${bit}/${model} \
                        --quant_file ${quantpath}/quantquik${bit}/${model} \
                        --n_ctx ${batch}  --n_batch $batch  --dataset_path ${dataset_path} --eval_accuracy True
   
                  
     
            fi         
        done 
done