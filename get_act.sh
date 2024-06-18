
CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
set -x

model=( Llama-2-7b )
model=( Aquila2-7b )
model=( Baichuan2-7b )
$CMD examples/smooth_quant_get_act.py  --model-name /mnt/octave/data/chenyidong/checkpoint/${model}  \
        --output-path /home/chenyidong/SC3/MixQ/src/act_scales/${model}.pt  --dataset-path /home/chenyidong/val.jsonl.zst 

