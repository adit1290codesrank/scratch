#include "../../../include/core/batchnorm_ops.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void bn_param_kernel(const float* X,float* mean,float* var,int N,int C,int spatial)
{
    int c=blockIdx.x;
    int tid=threadIdx.x;
    int stride=blockDim.x;
    
    float sum=0.0f,sum_sq=0.0f;
    for(int i=tid;i<N*spatial;i+=stride)
    {
        int n=i/spatial;
        int s=i%spatial;
        float val=X[n*(C*spatial)+c*spatial+s];
        sum+=val;
        sum_sq+=val*val;
    }
    
    __shared__ float s_sum[256],s_sq[256];
    s_sum[tid]=sum;
    s_sq[tid]=sum_sq;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1)
    {
        if(tid<s_step){s_sum[tid]+=s_sum[tid+s_step];s_sq[tid]+=s_sq[tid+s_step];}
        __syncthreads();
    }
    
    if(tid==0)
    {
        float m=s_sum[0]/(N*spatial);
        mean[c]=m;
        var[c]=(s_sq[0]/(N*spatial))-(m*m);
    }
}
void bn_param_gpu(const float* X,float* mean,float* var,int N,int C,int spatial){bn_param_kernel<<<C,256>>>(X,mean,var,N,C,spatial);}

__global__ void bn_forward_kernel(const float* X,const float* mean,const float* var,const float* gamma,const float* beta,float* Y,float* X_hat,int N,int C,int spatial,float eps)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N*C*spatial)
    {
        int c=(idx/spatial)%C;
        float x_hat=(X[idx]-mean[c])/sqrtf(var[c]+eps);
        if(X_hat!=nullptr) X_hat[idx]=x_hat;
        Y[idx]=gamma[c]*x_hat+beta[c];
    }
}
void bn_forward_gpu(const float* X,const float* mean,const float* var,const float* gamma,const float* beta,float* Y,float* X_hat,int N,int C,int spatial,float eps){
    int threads=256,blocks=(N*C*spatial+threads-1)/threads;
    bn_forward_kernel<<<blocks,threads>>>(X,mean,var,gamma,beta,Y,X_hat,N,C,spatial,eps);
}

__global__ void bn_running_kernel(float* r_m,float* r_v,const float* b_m,const float* b_v,float momentum,int C)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<C){r_m[c]=momentum*b_m[c]+(1.0f-momentum)*r_m[c];r_v[c]=momentum*b_v[c]+(1.0f-momentum)*r_v[c];}
}
void bn_running_gpu(float* r_m,float* r_v,const float* b_m,const float* b_v,float momentum,int C){
    int threads=256,blocks=(C+threads-1)/threads;
    bn_running_kernel<<<blocks,threads>>>(r_m,r_v,b_m,b_v,momentum,C);
}

__global__ void bn_backward_param_kernel(const float* dY,const float* X_hat,float* d_gamma,float* d_beta,int N,int C,int spatial)
{
    int c=blockIdx.x,tid=threadIdx.x,stride=blockDim.x;
    float dg=0.0f,db=0.0f;
    for(int i=tid;i<N*spatial;i+=stride)
    {
        int n=i/spatial,s=i%spatial,idx=n*(C*spatial)+c*spatial+s;
        dg+=dY[idx]*X_hat[idx];
        db+=dY[idx];
    }
    
    __shared__ float s_dg[256],s_db[256];
    s_dg[tid]=dg;
    s_db[tid]=db;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1)
    {
        if(tid<s_step){s_dg[tid]+=s_dg[tid+s_step];s_db[tid]+=s_db[tid+s_step];}
        __syncthreads();
    }
    
    if(tid==0){d_gamma[c]=s_dg[0];d_beta[c]=s_db[0];}
}
void bn_backward_param_gpu(const float* dY,const float* X_hat,float* d_gamma,float* d_beta,int N,int C,int spatial){bn_backward_param_kernel<<<C,256>>>(dY,X_hat,d_gamma,d_beta,N,C,spatial);}

__global__ void bn_backward_kernel(const float* dY,const float* X_hat,const float* var,const float* gamma,float* dX,const float* d_gamma,const float* d_beta,int N,int C,int spatial,float eps)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N*C*spatial)
    {
        int c=(idx/spatial)%C;
        float M=(float)(N*spatial);
        float inv_std=1.0f/sqrtf(var[c]+eps);
        dX[idx]=(gamma[c]*inv_std/M)*(M*dY[idx]-d_beta[c]-X_hat[idx]*d_gamma[c]);
    }
}
void bn_backward_gpu(const float* dY,const float* X_hat,const float* var,const float* gamma,float* dX,const float* d_gamma,const float* d_beta,int N,int C,int spatial,float eps){
    int threads=256,blocks=(N*C*spatial+threads-1)/threads;
    bn_backward_kernel<<<blocks,threads>>>(dY,X_hat,var,gamma,dX,d_gamma,d_beta,N,C,spatial,eps);
}