
for model in 7 13
do
for i in  1 2 4 6 8
do
python examples/benchmark.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size $i

done
done
