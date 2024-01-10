 
for len in 32 
	do
	for size in 7 
	do
		
		#CUDA_VISIBLE_DEVICES=0  python evalbitsand.py --model_path /mnt/data/zhongrx/Llama-2-${size}b-hf --quant_file /mnt/data/zhongrx/Llama-2-${size}b-hf  
		CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_path /mnt/data/zhongrx/Llama-2-${size}b-hf --quant_file /mnt/data/zhongrx/Llama-2-${size}b-hf  --quant_file /mnt/data/chenyd/Llama-2-${size}b-awq --model_type awq --n_ctx $len
		CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_path /mnt/data/zhongrx/Llama-2-${size}b-hf --quant_file /mnt/data/zhongrx/Llama-2-${size}b-hf  --quant_file /mnt/data/chenyd/Llama-2-${size}b-awq --model_type fp16 --n_ctx $len

		#CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_path /mnt/data/zhongrx/Llama-2-${size}b-hf --quant_file /mnt/data/zhongrx/Llama-2-${size}b-hf  --model_type bitsandbytesmix8   --n_ctx $len
		#CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_path /mnt/data/zhongrx/Llama-2-${size}b-hf --quant_file /mnt/data/zhongrx/Llama-2-${size}b-hf  --model_type bitsandbytesmix4   --n_ctx $len
	done
done

#CUDA_VISIBLE_DEVICES=0  python evalbitsand.py   --model_path   /mnt/data/huangkz/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546 --quant_file  /mnt/data/chenyd/models--facebook--opt-30b-awq  
#CUDA_VISIBLE_DEVICES=$1  python evalppl.py   --model_path   /mnt/data/huangkz/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546 --quant_file  /mnt/data/chenyd/models--facebook--opt-30b-awq  