

set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "
models=("Llama-2-70b")
type='fp16'




for model in "${models[@]}"
    do

    model_dir=/dataset/${model}
    output_dir=/code/checkpoint/checkpoint${type}/tllm_checkpoint_1gpu_fp16${model}
    engine_dir=/code/checkpoint/trt_engines${type}/tllm_checkpoint_1gpu_fp16${model}

    # CUDA_VISIBLE_DEVICES=4,5,6,7  http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
    # python  quantize.py --model_dir  ${model_dir}\
    # --output_dir ${output_dir}   --dtype float16  --pp_size 4

    # CUDA_VISIBLE_DEVICES=4,5,6,7 trtllm-build --checkpoint_dir ${output_dir} \
    #         --output_dir  ${engine_dir} \
    #         --gemm_plugin float16

    CUDA_VISIBLE_DEVICES=0,1,2,3 http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  mpirun -np 4 --allow-run-as-root   python  summarize.py --test_trt_llm \
            --hf_model_dir ${model_dir} \
            --data_type fp16 \
            --engine_dir ${engine_dir}
done 


