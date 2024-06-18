for i in {0..39}; do
    # echo "evaluate ${file}" >> eval_out
    srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py $i result/llama13b-1.csv LLama-2-13b q >>temp13
done

for i in {0..31}; do
    # echo "evaluate ${file}" >> eval_out
    srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py $i result/llama7b-1.csv LLama-2-7b q >>temp7
done

for i in {0..31}; do
    # echo "evaluate ${file}" >> eval_out
    srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py $i result/llama70b-1.csv LLama-2-70b q >>temp70
done

for i in {0..31}; do
    # echo "evaluate ${file}" >> eval_out
    srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py $i result/llama70b-down.csv LLama-2-70b down
done

for i in {0..31}; do
    # echo "evaluate ${file}" >> eval_out
    srun -N 1 --pty --gres=gpu:a100:1 -p octave -A public python benchbitsand.py $i result/llama70b-up.csv LLama-2-70b up
done
