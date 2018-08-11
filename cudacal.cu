#include <stdexcept>

__global__ void _cuda_vdSqr(int n, double *x, double *y, int run_num){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
    for (int j = 0; j < run_num; ++j) y[i] = x[i] * x[i];
  }
}

static void errChk(cudaError_t err){
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

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