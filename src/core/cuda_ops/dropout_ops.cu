#include "../../../include/core/dropout_ops.h"
#include "../../../include/core/dtype_ops.h"

__device__ float lcg_rand(unsigned int seed,int index)
{
    seed=(seed*1664525+1013904223)^index;
    seed=(seed*1664525+1013904223);
    return (float)(seed&0x00FFFFFF)/(float)0x01000000;
}

template<typename T>
__global__ void dropout_forward_kernel(const T* X,T* Y,T* mask,float p,int size,unsigned int seed)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size)
    {
        float rand_val=lcg_rand(seed,index);
        if(rand_val<p)
        {
            mask[index]=from_float<T>(0.0f);
            Y[index]=from_float<T>(0.0f);
        }
        else
        {
            float m=1.0f/(1.0f-p);
            mask[index]=from_float<T>(m);
            Y[index]=from_float<T>(to_float(X[index])*m);
        }
    }
}

// __global__ void dropout_forward_kernel(const float* X,float* Y,float* mask,float p,int size,unsigned int seed)
// {
//     int index=blockIdx.x*blockDim.x+threadIdx.x;
//     if(index<size)
//     {
//         float rand_val=lcg_rand(seed,index);
//         if(rand_val<p)
//         {
//             mask[index]=0.0f;
//             Y[index]=0.0f;
//         }
//         else
//         {
//             mask[index]=1.0f/(1.0f-p);
//             Y[index]=X[index]*mask[index];
//         }
//     }
// }

template<typename T>
__global__ void dropout_backward_kernel(const T* dY,T* dX,const T* mask,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) dX[index]=from_float<T>(to_float(dY[index])*to_float(mask[index]));
}

// __global__ void dropout_backward_kernel(const float* dY,float* dX,const float* mask,float p,int size)
// {
//     int index=blockIdx.x*blockDim.x+threadIdx.x;
//     if(index<size) dX[index]=dY[index]*mask[index];
// }

void dropout_forward_gpu(const Tensor& X,Tensor& Y,Tensor& mask,float p,unsigned int seed)
{
    int size=X.total_elements();
    int threads=256;
    int blocks=(size+threads-1)/threads;
    if(X.dtype()==DType::F32) dropout_forward_kernel<float><<<blocks,threads>>>(X.get_data(),Y.get_data(),mask.get_data(),p,size,seed);
    else dropout_forward_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)X.get_data(),(__nv_bfloat16*)Y.get_data(),(__nv_bfloat16*)mask.get_data(),p,size,seed);
}

void dropout_backward_gpu(const Tensor& dY,Tensor& dX,const Tensor& mask,float p)
{
    (void)p; 
    int size=dY.total_elements();
    int threads=256;
    int blocks=(size+threads-1)/threads;
    if(dY.dtype()==DType::F32)dropout_backward_kernel<float><<<blocks,threads>>>(dY.get_data(),dX.get_data(),mask.get_data(),size);
    else dropout_backward_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)dY.get_data(),(__nv_bfloat16*)dX.get_data(),(__nv_bfloat16*)mask.get_data(),size);
}

// void dropout_forward_gpu(const float* X,float* Y,float* mask,float p,int size,unsigned int seed)
// {
//     int threads=256;
//     int blocks=(size+threads-1)/threads;
//     dropout_forward_kernel<<<blocks,threads>>>(X,Y,mask,p,size,seed);
// }

// void dropout_backward_gpu(const float* dY,float* dX,const float* mask,float p,int size)
// {
//     int threads=256;
//     int blocks=(size+threads-1)/threads;
//     dropout_backward_kernel<<<blocks,threads>>>(dY,dX,mask,p,size);
// }
