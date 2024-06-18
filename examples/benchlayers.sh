#!/bin/bash

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama13b-bd-unschedule.csv LLama-2-13b q
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 20 result/llama13b-bd-unschedule.csv LLama-2-13b q
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 39 result/llama13b-bd-unschedule.csv LLama-2-13b q

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama7b-bd-unschedule.csv LLama-2-7b q
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama7b-bd-unschedule.csv LLama-2-7b q
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama7b-bd-unschedule.csv LLama-2-7b q

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama70b-bd-unschedule.csv LLama-2-70b q
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama70b-bd-unschedule.csv LLama-2-70b q
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama70b-bd-unschedule.csv LLama-2-70b q

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama7b-up-1.csv LLama-2-7b up
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama7b-up-1.csv LLama-2-7b up
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama7b-up-1.csv LLama-2-7b up

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama7b-down-1.csv LLama-2-7b down
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama7b-down-1.csv LLama-2-7b down
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama7b-down-1.csv LLama-2-7b down

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama70b-down-1.csv LLama-2-70b down
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama70b-down-1.csv LLama-2-70b down
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama70b-down-1.csv LLama-2-70b down

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama13b-down-1.csv LLama-2-13b down
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 20 result/llama13b-down-1.csv LLama-2-13b down
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 39 result/llama13b-down-1.csv LLama-2-13b down

srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama70b-up-1.csv LLama-2-70b up
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama70b-up-1.csv LLama-2-70b up
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama70b-up-1.csv LLama-2-70b up

