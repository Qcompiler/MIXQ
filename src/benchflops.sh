metrics="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"

# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics+="dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum"
#for batch in 1 2 4  8  16  32  64  128
for batch in 1 
    do
    for seq in   32 
        do
            for model in 7  
                do
            
                #CUDA_VISIBLE_DEVICES=$1 /home/chenyidong/nsight/ncu -k gemm_forward_4bit_cuda_m16n128k32 --set roofline -o outputncu/out_model_${model}_batch${batch}_seq${seq}.out}  python benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}  
                #CUDA_VISIBLE_DEVICES=$1 /home/chenyidong/nsight/ncu -k gemm_forward_4bit_cuda_m16n128k32 -c 10   --metrics ${metrics} --csv     python benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}   >  outputncu/out_model_${model}_batch${batch}_seq${seq}.csv
                #CUDA_VISIBLE_DEVICES=$1 nsys nvprof -o  outputnsys/out_batch${batch}_seq${seq} python benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}  
                #CUDA_VISIBLE_DEVICES=$1 python -m cProfile  -o  outputcprof/out_model_${model}_batch${batch}_seq${seq}.out   benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}   >  outputnsyscsv/out_model_${model}_batch${batch}_seq${seq}.csv
                CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}  
                CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_type fp16 --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}  

                #CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}   >  outputnsyscsv/out_model_${model}_batch${batch}_seq${seq}.csv
                #CUDA_VISIBLE_DEVICES=$1 python -m cProfile  -o  outputcproffp16/out_model_${model}_batch${batch}_seq${seq}.out   benchflops.py --model_type fp16 --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}   >  outputnsyscsvfp16/out_model_${model}_batch${batch}_seq${seq}.csv

            done
        done
done