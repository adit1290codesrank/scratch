#include "../../../include/core/gap_ops.h"
#include <cuda_runtime.h>

__global__ void gap_forward_kernel(const float* X,float* Y,int s,int total)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;

    if(index<total)
    {
        float sum=0.0f;
        for(int i=0;i<s;i++) sum+=X[index*s+i];
        Y[index]=sum/s;
    }
}

void gap_forward_gpu(const float* X,float* Y,int n,int c,int s)
{
    int total=n*c;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    gap_forward_kernel<<<blocks,threads>>>(X,Y,s,total);
}

__global__ void gap_backward_kernel(const float* dY,float* dX,int s,int total)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;

    if(index<total) dX[index]=dY[index/s]/s;
}

void gap_backward_gpu(const float* dY,float* dX,int n,int c,int s)
{
    int total=n*c*s;
    int threads=256;
    int blocks=(threads+total-1)/threads;
    gap_backward_kernel<<<blocks,threads>>>(dY,dX,s,total);
}