#include "../../../include/core/layernorm_ops.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void layernorm_forward_kernel(const float* X,const float* g,const float* b,float* Y,int n,int d,float eps)
{
    int r=blockIdx.x,t=threadIdx.x,s=blockDim.x;
    
    float sum=0.0f;
    for(int i=t;i<d;i+=s) sum+=X[r*d+i];
    
    __shared__ float s_sum[256];
    s_sum[t]=sum;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1)
    {
        if(t<s_step) s_sum[t]+=s_sum[t+s_step];
        __syncthreads();
    }
    
    __shared__ float s_mean;
    if(t==0) s_mean=s_sum[0]/d;
    __syncthreads();
    
    float mean=s_mean;
    float var_sum=0.0f;
    for(int i=t;i<d;i+=s)
    {
        float diff=X[r*d+i]-mean;
        var_sum+=diff*diff;
    }
    
    s_sum[t]=var_sum;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1)
    {
        if(t<s_step) s_sum[t]+=s_sum[t+s_step];
        __syncthreads();
    }
    
    __shared__ float s_var;
    if(t==0) s_var=s_sum[0]/d;
    __syncthreads();
    
    float inv_std=rsqrtf(s_var+eps);
    for(int i=t;i<d;i+=s)
    {
        int idx=r*d+i;
        Y[idx]=g[i]*((X[idx]-mean)*inv_std)+b[i];
    }
}

void layernorm_forward(const float* X,const float* g,const float* b,float* Y,int n,int d,float eps)
{
    int blocks=n;
    int threads=256;
    layernorm_forward_kernel<<<blocks,threads>>>(X,g,b,Y,n,d,eps);
}

__global__ void layernorm_backward_kernel(const float* dY,const float* X,const float* g,float* dg,float* db,float* dX,int n,int d,float eps)
{
    int r=blockIdx.x,t=threadIdx.x,s=blockDim.x;
    
    float sum=0.0f;
    for(int i=t;i<d;i+=s) sum+=X[r*d+i];
    
    __shared__ float s_sum[256];
    __shared__ float s_sum2[256];
    s_sum[t]=sum;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1){if(t<s_step) s_sum[t]+=s_sum[t+s_step];__syncthreads();}
    
    __shared__ float s_mean;
    if(t==0) s_mean=s_sum[0]/d;
    __syncthreads();
    
    float mean=s_mean;
    float var_sum=0.0f;
    for(int i=t;i<d;i+=s){float diff=X[r*d+i]-mean;var_sum+=diff*diff;}
    
    s_sum[t]=var_sum;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1){if(t<s_step) s_sum[t]+=s_sum[t+s_step];__syncthreads();}
    
    __shared__ float s_var;
    if(t==0) s_var=s_sum[0]/d;
    __syncthreads();
    
    float inv_std=rsqrtf(s_var+eps);
    float sum_dY=0.0f;
    float sum_dY_Xhat=0.0f;
    for(int i=t;i<d;i+=s)
    {
        float x_hat=(X[r*d+i]-mean)*inv_std;
        float dy=dY[r*d+i];
        sum_dY+=dy;
        sum_dY_Xhat+=dy*x_hat;
    }
    
    s_sum[t]=sum_dY;
    s_sum2[t]=sum_dY_Xhat;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1){if(t<s_step){s_sum[t]+=s_sum[t+s_step];s_sum2[t]+=s_sum2[t+s_step];}__syncthreads();}
    
    __shared__ float r_sum_dY;
    __shared__ float r_sum_dY_Xhat;
    if(t==0){r_sum_dY=s_sum[0];r_sum_dY_Xhat=s_sum2[0];}
    __syncthreads();
    
    for(int i=t;i<d;i+=s)
    {
        int idx=r*d+i;
        float x_hat=(X[idx]-mean)*inv_std;
        float dy=dY[idx];
        dX[idx]=(g[i]*inv_std)*(dy-(r_sum_dY/d)-x_hat*(r_sum_dY_Xhat/d));
        atomicAdd(&dg[i],dy*x_hat);
        atomicAdd(&db[i],dy);
    }
}

void layernorm_backward(const float* dY,const float* X,const float* g,float* dg,float* db,float* dX,int N,int d,float eps)
{
    cudaMemset(dg,0,d*sizeof(float));
    cudaMemset(db,0,d*sizeof(float));
    int blocks=N;
    int threads=256;
    layernorm_backward_kernel<<<blocks,threads>>>(dY,X,g,dg,db,dX,N,d,eps);
}
