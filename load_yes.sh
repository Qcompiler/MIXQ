source /opt/spack/share/spack/setup-env.sh



export CC=gcc
export CXX=g++



spack load cuda@12.1.1%gcc@11.3.0
#spack load cuda@11.8.0%gcc@11.3.0

spack load gcc@8.5.0
conda activate h100
