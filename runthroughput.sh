

if [ $2 == a100 ]
    then
    CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
    else
    CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
fi

export http_proxy=127.0.0.1:7890 
export https_proxy=127.0.0.1:7890
set -x

quantpath=/home/dataset/quant/quant
modelpath=/mnt/octave/data/chenyidong/checkpoint

for batch in   32 64 128 256 512
#for batch in  1  

    do
    for seq in   1024  
        do
            ##model_type=Aquila2
            #model_type=opt
            #model_type=Mistral
            #model_type=gpt-j
            #model_type=falcon
            model_type=$3
            
            
            
            data_types=( "mix"  )
            for bit in 4 8 
                do
                for data_type in "${data_types[@]}"
                    do
                    model=${model_type}
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD}  benchflops.py  --model_type ${data_type} --model_path  \
                    ${quantpath}${bit}/${model} \
                    --quant_file ${quantpath}${bit}/${model} \
                    --batch_size ${batch} --bit ${bit} --dataset_path /home/chenyidong/checkpoint/dataset
                done 
            done
            
            
            # data_types=( "quik"  )
            # for bit in  4
            #     do
            #     for data_type in "${data_types[@]}"
            #         do
            #         model=${model_type}
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD}  benchflops.py  --model_type ${data_type} --model_path  \
            #         ${quantpath}quik${bit}/${model} \
            #         --quant_file ${quantpath}quik${bit}/${model} \
            #         --batch_size ${batch} --bit ${bit} --dataset_path /home/chenyidong/checkpoint/dataset
            #     done 
            # done
            # data_types=(  "fp16"   )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
            #         /mnt/octave/data/chenyidong/checkpoint/${model} \
            #         --quant_file /mnt/octave/data/chenyidong/checkpoint/${model} --batch_size ${batch}


            #     done
            # done
            # data_types=( "awq"   )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
            #         /mnt/octave/data/chenyidong/checkpoint/awqquant/${model} \
            #         --quant_file /mnt/octave/data/chenyidong/checkpoint/awqquant/${model} --batch_size ${batch}
            #     done
            # done


         
        done 
done
