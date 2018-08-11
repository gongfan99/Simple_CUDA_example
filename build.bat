nvcc --gpu-architecture=sm_50 --device-c cudacal.cu
nvcc --gpu-architecture=sm_50 --device-link cudacal.obj --output-file cudacal_link.obj
cl /O2 /Qpar /Qvec-report:2 /I $(CudaToolkitDir)/include /EHsc /Tptest.cc cudacal.obj cudacal_link.obj cudart.lib /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64\" /out:test.exe