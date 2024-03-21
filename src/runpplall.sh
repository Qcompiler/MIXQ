

CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL "
export http_proxy=127.0.0.1:8892 
export https_proxy=127.0.0.1:8892
set -x

for batch in    512 
    do
    for seq in   64  
        do
            ##model_type=Aquila2
            #model_type=opt
            #model_type=Mistral
            #model_type=gpt-j
            #model_type=falcon
            model_type=Llama-2
            
            
            models=(  "Llama-2-7b" "Baichuan2-7b" "Baichuan2-13b" "Llama-65b"  "Llama-2-70b" "Aquila2-7b" "Aquila2-34b" falcon-7b "falcon-40b" "Mistral-7b")  
            models=(  "vicuna-33b-v1.3" "vicuna-7b-v1.5" )
            data_types=(  "fp16" "bitsandbytes"  )
            for data_type in "${data_types[@]}"
                do
                for model in "${models[@]}"
                    do
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                    python evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
                    /data/chenyidong/checkpoint/${model} \
                    --quant_file /data/chenyidong/checkpoint/${model} \
                    --n_ctx $batch --n_batch $batch  --eval_accuracy True


                done
            done
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
            # data_types=( "mix"  )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
            #         python evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /data/chenyidong/checkpoint/quant/${model} \
            #         --quant_file /data/chenyidong/checkpoint/quant/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True


            #     done
            # done

         
        done 
done
