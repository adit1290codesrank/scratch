#include "../../../include/core/dropout_ops.h"

__device__ float lcg_rand(unsigned int seed,int index)
{
    seed=(seed*1664525+1013904223)^index;
    seed=(seed*1664525+1013904223);
    return (float)(seed&0x00FFFFFF)/(float)0x01000000;
}

__global__ void dropout_fwd_kernel(const float* X,float* Y,float* mask,float p,int size,unsigned int seed)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size)
    {
        float rand_val=lcg_rand(seed,index);
        if(rand_val<p)
        {
            mask[index]=0.0f;
            Y[index]=0.0f;
        }
        else
        {
            mask[index]=1.0f/(1.0f-p);
            Y[index]=X[index]*mask[index];
        }
    }
}

__global__ void dropout_bwd_kernel(const float* dY,float* dX,const float* mask,float p,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) dX[index]=dY[index]*mask[index];
}

void dropout_forward_gpu(const float* X,float* Y,float* mask,float p,int size,unsigned int seed)
{
    int threads=256;
    int blocks=(size+threads-1)/threads;
    dropout_fwd_kernel<<<blocks,threads>>>(X,Y,mask,p,size,seed);
}

void dropout_backward_gpu(const float* dY,float* dX,const float* mask,float p,int size)
{
    int threads=256;
    int blocks=(size+threads-1)/threads;
    dropout_bwd_kernel<<<blocks,threads>>>(dY,dX,mask,p,size);
}
