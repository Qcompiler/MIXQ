if [ $2 == a100 ]
    then
    CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
fi

if [ $2 == h100 ]
    then
    CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
fi
set -x
data_types=$3
quantpath=/home/dataset/quant/quant
modelpath=/home/dataset
port=8892
models=(   $4 )
for batch in    32 
    do
    for seq in   64  
        do
            ##model_type=Aquila2
            #model_type=opt
            #model_type=Mistral
            #model_type=gpt-j
            #model_type=falcon
            # model_type=Llama-2
            
            

            # data_types=( "fp16" )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         ${modelpath}/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True --dataset_path /home/chenyidong/checkpoint/dataset
            #     done
            # done


            # models=(    "gpt-j-6b" )
            # data_types=( "awq"   )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
            #         python evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /data/chenyidong/checkpoint/awqquant/${model} \
            #         --quant_file /data/chenyidong/checkpoint/awqquant/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True
            #     done
            # done
            
            if [ ${data_types} == mix8 ]
                then
                    bit=8
                    for data_type in "${data_types[@]}"
                        do
                        for model in "${models[@]}"
                            do
                            rm -r ${quantpath}${bit}/${model}/*.safetensors
                            echo ${model}          
                            CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:${port} https_proxy=127.0.0.1:${port}  \
                            ${CMD} evalppl.py --model_type ${data_type} --model_path  \
                            ${quantpath}${bit}/${model} \
                            --quant_file  ${quantpath}${bit}/${model} \
                            --n_ctx $batch --n_batch $batch  --eval_accuracy True --dataset_path /home/dataset/quant/checkpoint/dataset
                        done
                    done
            fi
            # models=(    "Llama-2-7b" )
            # bit=4
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:${port} https_proxy=127.0.0.1:${port}  \
            #         ${CMD} evalppl.py --model_type ${data_type} --model_path  \
            #         ${quantpath}${bit}/${model} \
            #         --quant_file  ${quantpath}${bit}/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True --dataset_path /home/chenyidong/checkpoint/dataset
            #     done
            # done
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /home/dataset/quant${bit}/${model} \
            #         --quant_file  /home/dataset/quant${bit}/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True
            #     done
            # done
         
        done 
done
