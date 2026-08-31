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