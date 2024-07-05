

set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "
models=("Llama-2-1b")
for model in "${models[@]}"
    do
    CUDA_VISIBLE_DEVICES=0  http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
    python  quantize.py --model_dir /code/tensorrt_llm/manual_plugin/checkpoint/${model} \
    --output_dir /code/tensorrt_llm/manual_plugin/checkpoint/checkpoinweight/tllm_checkpoint_1gpu_fp16${model} \
                                   --dtype float16 \
                                   --qformat int4_awq  --calib_size 32

    # trtllm-build --checkpoint_dir /code/tensorrt_llm/manual_plugin/checkpoint/checkpoinmix/tllm_checkpoint_1gpu_fp16${model} \
    #         --output_dir /code/tensorrt_llm/manual_plugin/checkpoint/trt_enginesmix/tllm_checkpoint_1gpu_fp16${model} \
    #         --gemm_plugin float16 --mix_precision int8 \
    #         --strongly_typed
done 

