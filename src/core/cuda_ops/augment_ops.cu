#include "../../../include/core/augment_ops.h"
#include <cuda_runtime.h>

__global__ void augment_kernel(const float* X,float* Y,const AugmentParams* p,int b,int c,int h,int w,int total)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<total)
    {
        int wout=index%w,hout=(index/w)%h,cout=(index/(w*h))%c,n=index/(c*h*w);

        int w_=wout,h_=hout-p[n].dy;
        if(p[n].flip)w_=w-1-wout;
        w_-=p[n].dx;

        if(w_>=0 && w_<w && h_>=0 && h_<h) Y[index]=X[n*c*h*w+cout*h*w+h_*w+w_];
        else Y[index]=0.0f;
    }
}

void* alloc_augment(int b)
{
    void* ptr;
    cudaMalloc(&ptr,b*sizeof(AugmentParams));
    return ptr;
}

void free_augment(void* ptr){if(ptr) cudaFree(ptr);}

void augment_gpu(const float* X,float* Y,void* pgpu,const AugmentParams* p,int b,int c,int h,int w)
{
    cudaMemcpy(pgpu,p,b*sizeof(AugmentParams),cudaMemcpyHostToDevice);

    int total=b*c*h*w;
    int threads=256;
    int blocks=(threads+total-1)/threads;
    augment_kernel<<<blocks,threads>>>(X,Y,(AugmentParams*)pgpu,b,c,h,w,total);
}