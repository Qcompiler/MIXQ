set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "
models=("Llama-2-70b")

# pkill -9 python
# pkill -9 bash.sh
gpu=4
for model in "${models[@]}"
    do

    model_dir=/dataset/${model}
    quant_dir=/code/checkpoint/checkpoinmix/tllm_checkpoint_${gpu}gpu_fp16${model}
    out_dir=/code/checkpoint/trt_enginesmix/tllm_checkpoint_${gpu}gpu_fp16${model}

    # rm -r ${quant_dir}
    # rm -r ${out_dir}
    # CUDA_VISIBLE_DEVICES=4,5,6,7  http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
    # python  quantize.py --model_dir  ${model_dir} \
    # --output_dir  ${quant_dir}  --dtype float16 --device  cpu \
    #                                --qformat int8_mix  --calib_size 32 --pp_size ${gpu}

    # CUDA_VISIBLE_DEVICES=0,1,2,7 trtllm-build --checkpoint_dir ${quant_dir} \
    #    --output_dir ${out_dir} \
    #        --gemm_plugin float16 --mix_precision int8 

    CUDA_VISIBLE_DEVICES=0,1,2,7 http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
                 mpirun -np 4 --allow-run-as-root    python  summarize.py --test_trt_llm \
                       --hf_model_dir ${model_dir} \
                       --data_type fp16 \
                       --engine_dir ${out_dir}

done 

