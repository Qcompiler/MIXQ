set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "
models=("Llama-2-7b")

# pkill -9 python
# pkill -9 bash.sh
ngpu=1
for model in "${models[@]}"
    do

    model_dir=/code/checkpoint/${model}
    quant_dir=/code/checkpoint/checkpoinmix/tllm_checkpoint_${ngpu}gpu_fp16${model}
    out_dir=/code/checkpoint/trt_enginesmix/tllm_checkpoint_${ngpu}gpu_fp16${model}

    # rm -r ${quant_dir}
    # rm -r ${out_dir}
    # CUDA_VISIBLE_DEVICES=$1  http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
    # python  quantize.py --model_dir  ${model_dir} \
    # --output_dir  ${quant_dir}  --dtype float16 --device  cpu \
    #                                --qformat int8_mix  --calib_size 32 

    # CUDA_VISIBLE_DEVICES=$1 trtllm-build --checkpoint_dir ${quant_dir} \
    #    --output_dir ${out_dir} --max_input_len  2048 \
    #        --gemm_plugin float16 --mix_precision int8 

    CUDA_VISIBLE_DEVICES=$1 http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
                 python  summarize.py --test_trt_llm \
                       --hf_model_dir ${model_dir} \
                       --data_type fp16 \
                       --engine_dir ${out_dir}

done 

