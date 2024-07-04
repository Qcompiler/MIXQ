

if [ $2 == a100 ]
    then
    CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
fi

if [ $2 == h100 ]
    then
    CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
fi

export http_proxy=127.0.0.1:7890 
export https_proxy=127.0.0.1:7890
set -x

quantpath=/home/dataset/quant/quant
modelpath=/home/dataset

for batch in   32  
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
            
            
            
            # data_types=( "mix"  )
            # for bit in   8 
            #     do
            #     for data_type in "${data_types[@]}"
            #         do
            #         model=${model_type}
            #         echo ${model}   
            #         rm -r ${quantpath}${bit}/${model}/model.safetensors     
            #         CUDA_VISIBLE_DEVICES=$1    ${CMD}  benchflops.py  --model_type ${data_type} --model_path  \
            #         ${quantpath}${bit}/${model} \
            #         --quant_file ${quantpath}${bit}/${model} \
            #         --batch_size ${batch} --bit ${bit} --dataset_path /home/chenyidong/checkpoint/dataset
            #     done 
            # done
            
            
            # data_types=( "quik"  )
            # for bit in  4
            #     do
            #     for data_type in "${data_types[@]}"
            #         do
            #         model=${model_type}
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1  ${CMD}  benchflops.py  --model_type ${data_type} --model_path  \
            #         ${quantpath}quik${bit}/${model} \
            #         --quant_file ${quantpath}quik${bit}/${model} \
            #         --batch_size ${batch} --bit ${bit} --dataset_path /home/chenyidong/checkpoint/dataset
            #     done 
            # done
            data_types=(  "fp16"   "bitsandbytes"   )
            for data_type in "${data_types[@]}"
                do
                model=${model_type}
                     
                echo ${model}          
                CUDA_VISIBLE_DEVICES=$1 ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                ${modelpath}/${model} \
                --quant_file ${modelpath}/${model} --batch_size ${batch} --dataset_path /home/chenyidong/checkpoint/dataset

            done
            # data_types=( "awq"   )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}
            #         CUDA_VISIBLE_DEVICES=$1    ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
            #         ${quantpath}/awqquant/${model} \
            #         --quant_file ${quantpath}t/awqquant/${model} --batch_size ${batch}
            #     done
            # done


         
        done 
done
