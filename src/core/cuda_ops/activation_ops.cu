#include "../../../include/core/activation_ops.h"
#include "../../../include/core/dtype_ops.h"

template<typename T>
__global__ void relu_forward_kernel(T *Y,int total)
{
    int index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<total) if(to_float(Y[index])<0.0f) Y[index]=from_float<T>(0.0f);
}

// __global__ void relu_forward_kernel(float *Y,int total)
// {
//     int index=threadIdx.x+blockIdx.x*blockDim.x;
//     if(index<total)if(Y[index]<0.0f) Y[index]=0.0f;
// }

void relu_forward(Tensor& Y)
{
    int total=Y.total_elements();
    int threads=256;
    int blocks=(threads+total-1)/threads;
    if(Y.dtype()==DType::F32) relu_forward_kernel<float><<<blocks,threads>>>(Y.get_data(),total);
    else relu_forward_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)Y.get_data(),total);
}

// void relu_forward(Tensor& Y)
// {
//     int total=Y.total_elements();
//     int threads=256;
//     int blocks=(threads+total-1)/threads;
//     relu_forward_kernel<<<blocks,threads>>>(Y.get_data(),total);
// }

template<typename T>
__global__ void relu_backward_kernel(const T *dY,const T *cached_X,T *dX,int total)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<total) dX[index]=(to_float(cached_X[index])>0.0f?dY[index]:from_float<T>(0.0f));
}

// __global__ void relu_backward_kernel(const float *dY,const float *cached_X,float *dX,int total)
// {
//     int index=blockIdx.x*blockDim.x+threadIdx.x;
//     if(index<total) dX[index]=(cached_X[index]>0.0f?dY[index]:0.0f);
// }

void relu_backward(const Tensor& dY,const Tensor& cached_X,Tensor& dX)
{
    int total=dX.total_elements();
    int threads=256;
    int blocks=(threads+total-1)/threads;
    if(dX.dtype()==DType::F32) relu_backward_kernel<float><<<blocks,threads>>>(dY.get_data(),cached_X.get_data(),dX.get_data(),total);
    else relu_backward_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)dY.get_data(),(__nv_bfloat16*)cached_X.get_data(),(__nv_bfloat16*)dX.get_data(),total);
}

// void relu_backward(const Tensor& dY,const Tensor& cached_X,Tensor& dX)
// {
//     int total=dX.total_elements();
//     int threads=256;
//     int blocks=(threads+total-1)/threads;
//     relu_backward_kernel<<<blocks,threads>>>(dY.get_data(),cached_X.get_data(),dX.get_data(),total);
// }


template<typename T>
__global__ void softmax_forward_kernel(T *X,int rows,int cols)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<rows)
    {
        float max_=to_float(X[index*cols]);
        for(int i=0;i<cols;i++){float v=to_float(X[i+index*cols]); if(v>max_)max_=v;}

        float sum=0.0f;
        for(int i=0;i<cols;i++)
        {
            float e=expf(to_float(X[i+index*cols])-max_);
            X[i+index*cols]=from_float<T>(e);
            sum+=e;
        }
        for(int i=0;i<cols;i++) X[i+index*cols]=from_float<T>(to_float(X[i+index*cols])/sum);
    }
}

void softmax_forward(Tensor& X)
{
    int cols=X.shape.back(),rows=X.total_elements()/cols;
    int threads=256;
    int blocks=(rows+threads-1)/threads;
    if(X.dtype()==DType::F32) softmax_forward_kernel<float><<<blocks,threads>>>(X.get_data(),rows,cols);
    else softmax_forward_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)X.get_data(),rows,cols);
}