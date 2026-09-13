#include "../../../include/core/memory_ops.h"
#include "../../../include/core/dtype_ops.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <curand.h>

void* raw_device_malloc(size_t bytes)
{
    void *ptr=nullptr;
    cudaError_t err=cudaMalloc(&ptr, bytes);
    if(err!=cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    return ptr;
}

void raw_device_free(void* ptr){cudaFree(ptr);}

void raw_zero_malloc(void *ptr, size_t bytes)
{
    cudaError_t err=cudaMemset(ptr, 0, bytes);
    if(err!=cudaSuccess) throw std::runtime_error("cudaMemset failed");
}

template<typename T>
__global__ void one_kernel(T* ptr,int total)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<total) ptr[index]=from_float<T>(1.0f);
}

void raw_one_malloc(void *ptr, size_t count, DType dt)
{
    int total=(int)count;
    int threads=256;
    int blocks=(threads+total-1)/threads;
    if(dt==DType::F32) one_kernel<float><<<blocks,threads>>>((float*)ptr,total);
    else one_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)ptr,total);
}

void raw_copy_malloc(void *dest,const void *src,size_t bytes)
{
    cudaError_t err = cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}

void raw_randn(float* ptr,size_t count,float mean,float std)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
    curandGenerateNormal(gen,ptr,count,mean,std);
    curandDestroyGenerator(gen);
}

void raw_copy_from_host(void* dest, const void* src, size_t bytes)
{
    cudaError_t err=cudaMemcpy(dest,src,bytes,cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}

void raw_copy_to_host(void* dest,const void* src,size_t bytes)
{
    cudaError_t err=cudaMemcpy(dest,src,bytes,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}
