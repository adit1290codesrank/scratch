#include "../../../include/core/activation_ops.h"

__global__ void relu_forward_kernel(float *Y,int total)
{
    int index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<total)if(Y[index]<0.0f) Y[index]=0.0f;
}

void relu_forward(Tensor& Y)
{
    int total=Y.total_elements();
    int threads=256;
    int blocks=(threads+total-1)/threads;
    relu_forward_kernel<<<blocks,threads>>>(Y.get_data(),total);
}

__global__ void relu_backward_kernel(const float *dY,const float *cached_X,float *dX,int total)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<total) dX[index]=(cached_X[index]>0.0f?dY[index]:0.0f);
}

void relu_backward(const Tensor& dY,const Tensor& cached_X,Tensor& dX)
{
    int total=dX.total_elements();
    int threads=256;
    int blocks=(threads+total-1)/threads;
    relu_backward_kernel<<<blocks,threads>>>(dY.get_data(),cached_X.get_data(),dX.get_data(),total);
}

