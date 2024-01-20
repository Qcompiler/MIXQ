
source /opt/spack/share/spack/setup-env.sh


spack load gcc@9.4.0


 
#spack load cuda@11.4.2
#spack load intel-mkl@2020.4.304
export CC=gcc
export CXX=g++
conda activate cuda121

export PATH=/home/spack/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/libxml2-2.9.10-ri6jeyf7qdry5mulqxdwmx5owutk43zu/bin:/home/spack/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/xz-5.2.5-z354vla7kbv54mck2n274ab753q3ppvr/bin:/home/spack/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/libiconv-1.16-tfvgzgt57fsmlcjofbmqvghva7ywxjbz/bin:/home/spack/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/gcc-9.4.0-2nydfcada6e6c5utkcdezspb66rxxwbs/bin:/home/spack/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/gcc-9.4.0-2nydfcada6e6c5utkcdezspb66rxxwbs/bin:/opt/spack/bin:/home/chenyidong/.vscode-server/bin/0ee08df0cf4527e40edc9aa28f4b5bd38bbff2b2/bin/remote-cli:/home/chenyidong/.local/bin:/home/chenyidong/anaconda3/envs/cuda121/bin:/home/chenyidong/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:/home/chenyidong/anaconda3/bin:/home/chenyidong/anaconda3/bin:/data/chenyidong/cuda121/bin


spack load cmake@3.21.4
