#include "../../../include/core/tensor_ops.h"
#include "../../../include/core/context.h"
#include <stdexcept>

Tensor multiply(const Tensor& a,bool transA,const Tensor& b,bool transB)
{
    int m=transA?a.cols():a.rows();
    int k1=transA?a.rows():a.cols();
    int k2=transB?b.cols():b.rows();
    int n=transB?b.rows():b.cols();
    if(k1!=k2) throw std::invalid_argument("Inner dimensions must match");

    Tensor c({m,n});
    cublasHandle_t handle = Context::get_instance().get_cublas_handle();

    cublasOperation_t opA=transA?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t opB=transB?CUBLAS_OP_T:CUBLAS_OP_N;

    float alpha=1.0f,beta = 0.0f;
    cublasSgemm(
        handle,
        opB,
        opA,
        n, m, k1,
        &alpha,
        b.get_data(),b.cols(),
        a.get_data(),a.cols(),
        &beta,
        c.get_data(),c.cols()
    );
    return c;
}

Tensor add(const Tensor& a,const Tensor& b)
{
    if(a.shape!=b.shape) throw std::invalid_argument("Tensor shapes must match");

    size_t total=a.total_elements();

    Tensor c(a.shape);
    cublasHandle_t handle = Context::get_instance().get_cublas_handle();

    float alpha=1.0f,beta = 1.0f;
    cublasSgeam(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        total,1,
        &alpha,
        a.get_data(),total,
        &beta,
        b.get_data(),total,
        c.get_data(),total
    );
    return c;
}


__global__ void add_bias_kernel(float *Y,const float *b,int n,int m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int total=n*m;
    if(index<total) Y[index]+=b[index%m];
}

void add_bias(Tensor& Y,const Tensor& b)
{
    int n=Y.rows(),m=Y.cols();
    int total=n*m;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    add_bias_kernel<<<blocks,threads>>>(Y.get_data(),b.get_data(),n,m);
}

__global__ void sum_rows_kernel(const float *dY,float *db,int n,int m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<m)
    {
        float sum=0.0f;
        for(int i=0;i<n;i++) sum+=dY[i*m+index];
        db[index]=sum;
    }
}

Tensor sum_rows(const Tensor& dY)
{
    int n=dY.rows(),m=dY.cols();
    Tensor db({1,m});
    int threads=256;
    int blocks=(m+threads-1)/threads;
    sum_rows_kernel<<<blocks,threads>>>(dY.get_data(),db.get_data(),n,m);
    return db;
}

__global__ void add_bias_conv_kernel(float *Y,const float *b,int batch,int cout,int s)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    int total=batch*cout*s;
    if(index<total) Y[index]+=b[(index/s)%cout];
}

void add_bias_conv(Tensor& Yflat,const Tensor& b)
{
    int batch=Yflat.shape[0],cout=Yflat.shape[1],s=Yflat.shape[2];
    int total=batch*cout*s;
    int threads=256;
    int blocks=(threads+total-1)/threads;
    add_bias_conv_kernel<<<blocks,threads>>>(Yflat.get_data(),b.get_data(),batch,cout,s);
}

__global__ void sum_spatial_kernel(const float *dYflat,float *db,int batch,int cout,int s)
{
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c<cout)
    {
        float sum=0.0f;
        for(int n=0;n<batch;n++)
        {
            for(int i=0;i<s;i++) sum+=dYflat[n*(cout*s)+c*s+i];
        }
        db[c]=sum;
    }
}

Tensor sum_spatial(const Tensor& dYflat)
{
    int batch=dYflat.shape[0],cout=dYflat.shape[1],s=dYflat.shape[2];
    Tensor db({1,cout});
    int threads=256;
    int blocks=(cout+threads-1)/threads;
    sum_spatial_kernel<<<blocks,threads>>>(dYflat.get_data(),db.get_data(),batch,cout,s);
    return db;
}

Tensor multiply_conv_forward(const Tensor& W,const Tensor& Xcol)
{
    int batch=Xcol.shape[0],fan_in=Xcol.shape[1],spatial=Xcol.shape[2],cout=W.shape[0];
    Tensor Yflat({batch,cout,spatial});
    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=0.0f;
    for(int n=0;n<batch;n++) cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,spatial,cout,fan_in,&alpha,Xcol.get_data()+n*(fan_in*spatial),spatial,W.get_data(),fan_in,&beta,Yflat.get_data()+n*(cout*spatial),spatial);
    return Yflat;
}

Tensor multiply_conv_backward_dX(const Tensor& W,const Tensor& dYflat)
{
    int batch=dYflat.shape[0],cout=dYflat.shape[1],spatial=dYflat.shape[2],fan_in=W.shape[1];
    Tensor dXcol({batch,fan_in,spatial});
    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=0.0f;
    for(int n=0;n<batch;n++) cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,spatial,fan_in,cout,&alpha,dYflat.get_data()+n*(cout*spatial),spatial,W.get_data(),fan_in,&beta,dXcol.get_data()+n*(fan_in*spatial),spatial);
    return dXcol;
}

void multiply_conv_backward_dW(Tensor& dW,const Tensor& dYflat,const Tensor& Xcol)
{
    int batch=dYflat.shape[0],cout=dYflat.shape[1],spatial=dYflat.shape[2],fan_in=Xcol.shape[1];
    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=1.0f;
    cudaMemset(dW.get_data(),0,cout*fan_in*sizeof(float));
    for(int n=0;n<batch;n++) cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,fan_in,cout,spatial,&alpha,Xcol.get_data()+n*(fan_in*spatial),spatial,dYflat.get_data()+n*(cout*spatial),spatial,&beta,dW.get_data(),fan_in);
}