# Simple_CUDA_example
A C++ example to use CUDA for Windows.

There are two steps in general.

The first step is to use Nvidia's compiler nvcc to compile/link the .cu file into two .obj files.

The second step is to use MSVC to compile the main C++ program and then link with the two .obj files created in the first step. 

# Build example
```c
// on Windows with VS 2015 and CUDA Toolkit 9.1 installed
git clone https://github.com/gongfan99/Simple_CUDA_example.git
build
test
```

# Reference
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html