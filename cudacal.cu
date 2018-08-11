/**
    cudacal.h
    Purpose: a simple CUDA example

    @author Fan Gong
    @version 1.0 07/03/18 
*/

#include <stdexcept>

/**
    device code to calculate sqr of x.
    In CUDA, device code is prefixed with "__device__", which only runs in GPU. It can only be called by other device code or kernel code.
	Sometimes, the code prefixed with both "__device__" and "__host__" can be called by device code, kernel code or host code.

    @param x The input number.
    @return The square of the input
*/
__device__ static double _cuda_sqr(double x){
  return x * x;
}

/**
    kernel code to calculate sqr of x array.
    In CUDA, kernal code is prefixed with "__global__", which can be called from host code in the form of "<<<blocksPerGrid, threadsPerBlock>>>"

    @param n The length of x and y.
    @param x The input array.
    @param y The output array.
    @param run_num The number of iterations.
    @return none
*/
__global__ void _cuda_vdSqr(int n, double *x, double *y, int run_num){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
    for (int j = 0; j < run_num; ++j) y[i] = _cuda_sqr(x[i]);
  }
}

/**
    error check function.

    @param err The return value of a CUDA function.
    @return none
*/
static void errChk(cudaError_t err){
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

/**
    host code.
    In CUDA, host code is the code that runs in the CPU, which calls the kernel code to do GPU calculation.
	It can be prefixed by "__host__" which is unnecessary since it is the default.

    @param n The length of x and y.
    @param x The input array.
    @param y The output array.
    @param run_num The number of iterations.
    @param threadsPerBlock The threads in each block. 256 is a good start.
    @param blocksPerGrid The blocks in each grid.
    @return none
*/
void cuda_vdSqr(int n, double *x, double *y, int run_num, int threadsPerBlock, int blocksPerGrid){
  size_t size = n * sizeof(double);
  double *d_x = nullptr;
  double *d_y = nullptr;
  errChk(cudaMalloc((void**)&d_x, size));
  errChk(cudaMalloc((void**)&d_y, size));
  
  errChk(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
  _cuda_vdSqr<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_y, run_num);
  
  errChk(cudaGetLastError());
  errChk(cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost));
  
  errChk(cudaFree(d_x));
  errChk(cudaFree(d_y));
} 