

CMD=" srun  -N 1 --pty --gres=gpu:a100:2 -p octave -A public python "
export http_proxy=127.0.0.1:7890 
export https_proxy=127.0.0.1:7890
set -x


bit=4
for batch in   512
#for batch in  1  

    do
    for seq in   64  
        do
            ##model_type=Aquila2
            #model_type=opt
            #model_type=Mistral
            #model_type=gpt-j
            #model_type=falcon
            model_type=Llama-2
            
            
            models=(     "Llama-2-7b"  ) 
            data_types=( "mix"  )
            
            for data_type in "${data_types[@]}"
                do
                for model in "${models[@]}"
                    do
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD} benchlatency.py  --model_type ${data_type} --model_path  \
                    /home/dataset/quant${bit}/${model} \
                    --quant_file /home/dataset/quant${bit}/${model} \
                    --batch_size ${batch} --bit ${bit}

                done
            done 
            
            data_types=(  "fp16"  , "bitsandbytes" )
            for data_type in "${data_types[@]}"
                do
                for model in "${models[@]}"
                    do
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                    /mnt/octave/data/chenyidong/checkpoint/${model} \
                    --quant_file /mnt/octave/data/chenyidong/checkpoint/${model} --batch_size ${batch}


                done
            done
            data_types=( "awq"   )
            for data_type in "${data_types[@]}"
                do
                for model in "${models[@]}"
                    do
                    echo ${model}
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                    /mnt/octave/data/chenyidong/checkpoint/awqquant/${model} \
                    --quant_file /mnt/octave/data/chenyidong/checkpoint/awqquant/${model} --batch_size ${batch}
                done
            done


         
        done 
done
