#include "../../../include/core/loss_ops.h"
#include <cuda_runtime.h>
#include "../../../include/core/dtype_ops.h"

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

template<typename T>
__global__ void sparse_ce_loss_kernel(const T* pred,const float* targets,float* loss,int* count,int batch_seq,int vocab,int ignore_index)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<batch_seq)
    {
        int cls=(int)targets[row];
        if(cls!=ignore_index)
        {
            atomicAdd(loss,-logf(to_float(pred[row*vocab+cls])+1e-7f));
            atomicAdd(count,1);
        }
    }
}

// __global__ void sparse_ce_loss_kernel(const float* pred,const float* targets,float* loss,int batch_seq,int vocab,int ignore_index)
// {
//     int row=blockIdx.x*blockDim.x+threadIdx.x;
//     if(row<batch_seq)
//     {
//         int cls=(int)targets[row];
//         if(cls!=ignore_index) atomicAdd(loss,-logf(pred[row*vocab+cls]+1e-7f)/(float)batch_seq);
//     }
// }

template<typename T>
__global__ void sparse_ce_backward_kernel(const T* pred,const float* targets,T* dY,int batch_seq,int vocab,int ignore_index,int valid_count)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<batch_seq*vocab)
    {
        int row=idx/vocab,col=idx%vocab;
        int cls=(int)targets[row];
        if(cls!=ignore_index) dY[idx]=from_float<T>((to_float(pred[idx])-(col==cls?1.0f:0.0f))/(float)valid_count);
        else dY[idx]=from_float<T>(0.0f);
    }
}

// __global__ void sparse_ce_backward_kernel(const float* pred,const float* targets,float* dY,int batch_seq,int vocab,int ignore_index)
// {
//     int idx=blockIdx.x*blockDim.x+threadIdx.x;
//     if(idx<batch_seq*vocab)
//     {
//         int row=idx/vocab,col=idx%vocab;
//         int cls=(int)targets[row];
//         if(cls!=ignore_index) dY[idx]=(pred[idx]-(col==cls?1.0f:0.0f))/(float)batch_seq;
//         else dY[idx]=0.0f;
//     }
// }

float sparse_ce_forward(const Tensor& pred,const Tensor& targets,int ignore_index,int* out_valid_count)
{
    int batch_seq=pred.shape[0]*(pred.shape.size()>2?pred.shape[1]:1);
    int vocab=pred.shape.back();
    float* d_loss;
    int* d_count;
    cudaMalloc(&d_loss,sizeof(float));
    cudaMemset(d_loss,0,sizeof(float));
    cudaMalloc(&d_count,sizeof(int));
    cudaMemset(d_count,0,sizeof(int));

    int threads=256;
    int blocks=(batch_seq+threads-1)/threads;
    if(pred.dtype()==DType::F32) sparse_ce_loss_kernel<float><<<blocks,threads>>>(pred.get_data(),targets.get_data(),d_loss,d_count,batch_seq,vocab,ignore_index);
    else sparse_ce_loss_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)pred.get_data(),targets.get_data(),d_loss,d_count,batch_seq,vocab,ignore_index);

    float h_loss=0.0f;
    int h_count=0;
    cudaMemcpy(&h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    cudaFree(d_count);
    if(out_valid_count) *out_valid_count=h_count;
    return h_count>0?h_loss/(float)h_count:0.0f;
}

// float sparse_ce_forward(const Tensor& pred,const Tensor& targets,int ignore_index)
// {
//     int batch_seq=pred.shape[0]*(pred.shape.size()>2?pred.shape[1]:1);
//     int vocab=pred.shape.back();
//     float* d_loss;
//     cudaMalloc(&d_loss,sizeof(float));
//     cudaMemset(d_loss,0,sizeof(float));

//     int threads=256;
//     int blocks=(batch_seq+threads-1)/threads;
//     sparse_ce_loss_kernel<<<blocks,threads>>>(pred.get_data(),targets.get_data(),d_loss,batch_seq,vocab,ignore_index);

//     float h_loss=0.0f;
//     cudaMemcpy(&h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
//     cudaFree(d_loss);
//     return h_loss;
// }

void sparse_ce_backward(const Tensor& pred,const Tensor& targets,Tensor& dY,int ignore_index,int valid_count)
{
    int batch_seq=pred.shape[0]*(pred.shape.size()>2?pred.shape[1]:1);
    int vocab=pred.shape.back();
    if(valid_count<=0) valid_count=batch_seq;
    int total=batch_seq*vocab;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    if(pred.dtype()==DType::F32) sparse_ce_backward_kernel<float><<<blocks,threads>>>(pred.get_data(),targets.get_data(),dY.get_data(),batch_seq,vocab,ignore_index,valid_count);
    else sparse_ce_backward_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)pred.get_data(),targets.get_data(),(__nv_bfloat16*)dY.get_data(),batch_seq,vocab,ignore_index,valid_count);
}

// void sparse_ce_backward(const Tensor& pred,const Tensor& targets,Tensor& dY,int ignore_index)
// {
//     int batch_seq=pred.shape[0]*(pred.shape.size()>2?pred.shape[1]:1);
//     int vocab=pred.shape.back();
//     int total=batch_seq*vocab;
//     int threads=256;
//     int blocks=(total+threads-1)/threads;
//     sparse_ce_backward_kernel<<<blocks,threads>>>(pred.get_data(),targets.get_data(),dY.get_data(),batch_seq,vocab,ignore_index);
// }