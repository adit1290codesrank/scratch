#include "../../../include/core/loss_ops.h"
#include <cuda_runtime.h>

__global__ void mse_loss_kernel(const float* pred,const float* target,float* loss, int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) atomicAdd(loss,((pred[index]-target[index])*(pred[index]-target[index]))/(float)size);
}

__global__ void mse_backward_kernel(const float* pred,const float* target,float* dY,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index < size) dY[index]=2.0f*(pred[index]-target[index])/(float)size;
}

float mse_forward(const Tensor& pred, const Tensor& target)
{
    int total = pred.total_elements();
    float* loss;
    cudaMalloc(&loss,sizeof(float));
    cudaMemset(loss,0,sizeof(float));

    int threads=256;
    int blocks=(total+threads-1)/threads;
    mse_loss_kernel<<<blocks,threads>>>(pred.get_data(),target.get_data(),loss,total);

    float h_loss=0.0f;
    cudaMemcpy(&h_loss,loss,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(loss);
    return h_loss;
}

void mse_backward(const Tensor& pred,const Tensor& target,Tensor& dY)
{
    int total=pred.total_elements();
    int threads=256;
    int blocks=(total+threads-1)/threads;
    mse_backward_kernel<<<blocks,threads>>>(pred.get_data(),target.get_data(),dY.get_data(),total);
}

__global__ void ce_loss_kernel(const float* pred,const float* target,float* loss,int size,int batch_size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index<size) atomicAdd(loss,-target[index]*logf(pred[index]+1e-7f)/(float)batch_size); 
}

__global__ void ce_backward_kernel(const float* pred,const float* target,float* dY,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index < size) dY[index] = pred[index] - target[index];
}

float ce_forward(const Tensor& pred,const Tensor& target)
{
    int total=pred.total_elements();
    int batch_size=pred.rows();
    float* d_loss;
    cudaMalloc(&d_loss,sizeof(float));
    cudaMemset(d_loss,0,sizeof(float));

    int threads=256;
    int blocks=(total+threads-1)/threads;
    ce_loss_kernel<<<blocks, threads>>>(pred.get_data(),target.get_data(),d_loss,total,batch_size);

    float h_loss=0.0f;
    cudaMemcpy(&h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss;
}

void ce_backward(const Tensor& pred,const Tensor& target,Tensor& dY)
{
    int total=pred.total_elements();
    int threads=256;
    int blocks=(total+threads-1)/threads;
    ce_backward_kernel<<<blocks,threads>>>(pred.get_data(),target.get_data(),dY.get_data(), total);
}


__global__ void ls_ce_loss_kernel(const float* pred,const float* target,float* loss,int size,int batch_size,int n,float a)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index<size) 
    {
        float s=target[index]*(1.0-a)+(a/n);
        atomicAdd(loss,-s*logf(pred[index]+1e-7f)/(float)batch_size); 
    }
}

__global__ void ls_ce_backward_kernel(const float* pred,const float* target,float* dY,int size,int n,float a)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index < size)
    { 
        float s=target[index]*(1.0-a)+(a/n);
        dY[index] = pred[index] - s;
    }
}

float ls_ce_forward(const Tensor& pred,const Tensor& target,int n,float a)
{
    int total=pred.total_elements();
    int batch_size=pred.rows();
    float* d_loss;
    cudaMalloc(&d_loss,sizeof(float));
    cudaMemset(d_loss,0,sizeof(float));

    int threads=256;
    int blocks=(total+threads-1)/threads;
    ls_ce_loss_kernel<<<blocks,threads>>>(pred.get_data(),target.get_data(),d_loss,total,batch_size,n,a);

    float h_loss=0.0f;
    cudaMemcpy(&h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss;
}

void ls_ce_backward(const Tensor& pred,const Tensor& target,Tensor& dY,int n,float a)
{
    int total=pred.total_elements();
    int threads=256;
    int blocks=(total+threads-1)/threads;
    ls_ce_backward_kernel<<<blocks,threads>>>(pred.get_data(),target.get_data(),dY.get_data(),total,n,a);
}