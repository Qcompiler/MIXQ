

#CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
#CMD="python "
CMD="srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python"
export http_proxy=127.0.0.1:7890 
export https_proxy=127.0.0.1:7890
set -x
bit=4
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
            
            
            # models=(  "Llama-2-7b" "Baichuan2-7b" "Baichuan2-13b" "Llama-65b"  "Llama-2-70b" "Aquila2-7b" "Aquila2-34b" falcon-7b "falcon-40b" "Mistral-7b")  
            # models=(    "opt-6.7b" )
            # data_types=( "fp16" )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /mnt/octave/data/chenyidong/checkpoint/${model} \
            #         --quant_file /mnt/octave/data/chenyidong/checkpoint/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True


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
            data_types=( "mix"  )
            models=(    "falcon-7b"  )
            models=(    "Llama-2-7b" )
           # models=(    "opt-6.7b" )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /home/dataset/quant8/${model} \
            #         --quant_file  /home/dataset/quant8/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True
            #     done
            # done

            for data_type in "${data_types[@]}"
                do
                for model in "${models[@]}"
                    do
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
                    /home/dataset/quant${bit}/${model} \
                    --quant_file  /home/dataset/quant${bit}/${model} \
                    --n_ctx $batch --n_batch $batch  --eval_accuracy True
                done
            done
         
        done 
done
