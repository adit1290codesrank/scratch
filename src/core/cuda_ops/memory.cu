#include "../../../include/core/memory_ops.h"
#include <cuda_runtime.h>
#include <stdexcept>

float* raw_device_malloc(size_t bytes)
{
    float *ptr=nullptr;
    cudaError_t err=cudaMalloc(&ptr, bytes);
    if(err!=cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    return ptr;
}

void raw_device_free(float* ptr){cudaFree(ptr);}

void raw_zero_malloc(float *ptr, size_t bytes)
{
    cudaError_t err=cudaMemset(ptr, 0, bytes);
    if(err!=cudaSuccess) throw std::runtime_error("cudaMemset failed");
}

void raw_copy_malloc(float *dest,float *src,size_t bytes)
{
    cudaError_t err = cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}
