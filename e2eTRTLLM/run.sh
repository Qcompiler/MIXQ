
set -x


model=Llama-2-7b
ngpu=1
type=fp16

if [ $1 == fp16 ]
  then
    model_dir=/code/checkpoint/${model}
    output_dir=/code/checkpoint/checkpoint${type}/tllm_checkpoint_1gpu_fp16${model}
    engine_dir=/code/checkpoint/trt_engines${type}/tllm_checkpoint_1gpu_fp16${model}
  else
     model_dir=/code/checkpoint/${model}
    quant_dir=/code/checkpoint/checkpoinmix/tllm_checkpoint_${ngpu}gpu_fp16${model}
     engine_dir=/code/checkpoint/trt_enginesmix/tllm_checkpoint_${ngpu}gpu_fp16${model}
fi

python3 run.py --engine_dir   ${engine_dir}  --use_py_session --no_add_special_tokens --max_output_len 100 \
--tokenizer_dir ${model_dir} --input_text "美国的首都在哪里? \n答案:"  --top_p 0.5 --top_k 0 --max_output_len 10