
for i in 1 2 4
do
python examples/benchmark.py --model_path   /mnt/data/huangkz/huggingface/hub/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546 --quant_file  /mnt/data/chenyd/models--facebook--opt-30b-awq  --batch_size $i
done
