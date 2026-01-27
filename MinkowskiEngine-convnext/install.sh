export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

CXX=g++-11 CC=gcc-11 python setup.py install --force_cuda --blas=openblas