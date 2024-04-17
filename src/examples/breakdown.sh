srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 0 result/llama70b-up-1.csv LLama-2-70b up
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 15 result/llama70b-up-1.csv LLama-2-70b up
srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py 31 result/llama70b-up-1.csv LLama-2-70b up
