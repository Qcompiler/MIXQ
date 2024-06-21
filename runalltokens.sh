set -x
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchalltokens.py --model_type mix --model_path /home/dataset/quant/quant8/Llama-2-7b --quant_file /home/dataset/quant/quant8/Llama-2-7b --batch_size 512 --bit 8 --dataset_path /home/chenyidong/checkpoint/dataset
