# cuda-uvm-gpt2

This repo evaluates the performance of [PyTorch-UVM](https://github.com/kooyunmo/pytorch-uvm/tree/53e458826f1895ab92c7b31a1c66fa60c29f84cd) with extremely large-scale language models (e.g. GPT-2, GPT-3). PyTorch-UVM adopts CUDA Unified Virtual Memory (a.k.a. UVM) to serve memory-intensive models with preventing the program execution from OOM by up to CPU memory capacity. UVM makes both CPU and GPU share the same virtual address space. Therefore, even though the GPU memory is physically oversubscribed (vanilla PyTorch occurs OOM in this case), victim memory block is implicitly migrated to CPU physical memory space without any explicit data off-loading command. The evicted data can be migrated to GPU memory again when it is on-demand.

## How to Build PyTorch-UVM

### Prerequisites
- Ubuntu 18.04
- anaconda3
- cuda-11.0
- cudnn 8.0.4 for cuda-11.0
- correct environment variables
``` bash
git clone --recursive https://github.com/kooyunmo/cuda-uvm-gpt2
cd cuda-uvm-gpt2/pytorch-uvm
git checkout uvm

# create new conda environment
conda create -n uvm-pytorch python=3.8 -y
conda activate uvm-pytorch

# environment variables, we need this setting for every installation and experiment
export CUDA_HOME=<YOUR_CUDA_11.0_PATH>
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDNN_LIB_DIR=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include/
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# install dependencies
# ensure prerequisites for pytorch build
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install -c pytorch magma-cuda110 -y

# install onnx
conda install -c conda-forge onnx -y

# downgrade protobuf ([why?](https://github.com/onnx/onnx/issues/2434))
conda install -c conda-forge protobuf=3.9 -y

# ensure prerequisites for caffe2 build
pip install future

# run setup.py
BUILD_TEST=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_NUMA=0 USE_MPI=0 python setup.py install
``` 

## Evaluate
``` bash
# install requirements
pip install -r requirements.txt

# run inference
CUDA_VISIBLE_DEVICES=<GPU_ID>
python run_gpt2.py \
    --model <MODEL_NAME> \
    --enable-prefetch \
    --enable-cudnn-benchmark \
    --num-streams <NUM_STREAMS> \
    --warmups <NUM_WARMUP_STEP>
```

