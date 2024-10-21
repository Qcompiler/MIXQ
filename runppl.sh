if [ $2 == a100 ]
    then
    CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
    CMD="  python "
fi

if [ $2 == h100 ]
    then
    CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
fi
set -x
data_types=$3
quantpath=/home/cyd/mixqdata/quant
modelpath=/home/cyd/mixqdata
port=8892
models=(   $4 )
for batch in    512 
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
            if [ ${data_types} == awq ]
                then
                    for data_type in "${data_types[@]}"
                        do
                        for model in "${models[@]}"
                            do
                            echo ${model}
                            CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                            python evalppl.py  --model_type ${data_type} --model_path  \
                            ${modelpath}/${model} \
                            --quant_file ${modelpath}/${model}-AWQ \
                            --n_ctx $batch --n_batch $batch  --eval_accuracy True
                        done
                    done
            fi
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
                            --n_ctx $batch --n_batch $batch  --eval_accuracy True 
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
