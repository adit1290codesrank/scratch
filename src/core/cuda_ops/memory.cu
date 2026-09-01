#include "../../../include/core/memory_ops.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <curand.h>

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

__global__ void one_kernel(float* ptr,int total)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<total) ptr[index]=1.0f;
}

void raw_one_malloc(float *ptr, size_t bytes)
{
    int total=bytes/sizeof(float);
    int threads=256;
    int blocks=(threads+total-1)/threads;
    one_kernel<<<blocks,threads>>>(ptr,total);
}

void raw_copy_malloc(float *dest,float *src,size_t bytes)
{
    cudaError_t err = cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}

void raw_randn(float* ptr,size_t size,float mean,float std)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
    curandGenerateNormal(gen,ptr,size,mean,std);
    curandDestroyGenerator(gen);
}

void raw_copy_from_host(float* dest, const float* src, size_t bytes)
{
    cudaError_t err=cudaMemcpy(dest,src,bytes,cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}

void raw_copy_to_host(float* dest,const float* src,size_t bytes)
{
    cudaError_t err=cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}