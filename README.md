# Simple_CUDA_example
A C++ example to use CUDA for Windows.

There are two steps to compile the CUDA code in general.

The first step is to use Nvidia's compiler nvcc to compile/link the .cu file into two .obj files.

The second step is to use MSVC to compile the main C++ program and then link with the two .obj files created in the first step. 

# Build example
```c
// on Windows with VS 2015 and CUDA Toolkit 9.1 installed
git clone https://github.com/gongfan99/Simple_CUDA_example.git
build
test
```
# Introduction
A NVIDIA GPU consists of several Streaming Multiprocessors (SMs). When GPU runs, each SM can process several blocks each of which consists of many threads. After each SM finishes processing the current batch of blocks, it can start to process the next batch of blocks. Generally the total number of threads that each SM processes concurrently can be several hundread.

The CUDA code is contained in .cu files. Each .cu file consists of three types of functions:

1. host code

The host code runs in CPU and is prefixed with "\__host__". The prefix can be omitted since "host code" is the default.

2. kernel code

The kernel code runs in GPU and is prefixed with "\__global__", which can only be called from host code in the form of "<<<blocksPerGrid, threadsPerBlock>>>"

3. device code

The device code running in GPU is simply the helper function that the kernel code or other device code can call.

Sometimes, the code can be prefixed with both "\__device__" and "\__host__", which make it callable by device code, kernel code and host code.

# Reference
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html