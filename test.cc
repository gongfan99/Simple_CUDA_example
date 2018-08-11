#include <windows.h>

extern void cuda_vdSqr(int n, double *x, double *y, int run_num, int threadsPerBlock, int blocksPerGrid);

int main(int argc, char **argv){
  constexpr int run_num = 10000;
  constexpr int n = 500000;
  constexpr int threadsPerBlock = 256;
  const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  
  double *x = new double[n];
  double *y = new double[n];
  
  for (int i = 0; i < n; ++i) x[i] = rand();
  
  cuda_vdSqr(n, x, y, run_num, threadsPerBlock, blocksPerGrid);
  
  delete [] x;
  delete [] y;
}