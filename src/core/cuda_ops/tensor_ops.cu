__global__ void padding_mask_kernel(float* S,const float* mask,int heads,int seq_len)
{
    int row=blockIdx.x,b=row/(heads*seq_len),t=threadIdx.x,s=blockDim.x;
    for(int i=t;i<seq_len;i+=s) if(mask[b*seq_len+i]==0.0f) S[row*seq_len+i]=-1e9f;
}
#include "../../../include/core/tensor_ops.h"
#include "../../../include/core/context.h"
#include <cuda_runtime.h>
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
extern void softmax_forward(Tensor& X);

void attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Out,int heads,Tensor& S,const Tensor* mask)
{
    int batch_seq=Q.rows(),dmodel=Q.cols();
    
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    
    int dimension=dmodel/heads;
    
    S=Tensor::zeros({batch,heads,seq_len,seq_len});
    
    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f/sqrtf((float)dimension);float beta=0.0f;
    
    for(int i=0;i<batch;i++) 
    {
        float* batch_Q=Q.get_data()+(i*seq_len*dmodel),*batch_K=K.get_data()+(i*seq_len*dmodel),*batch_S=S.get_data()+(i*heads*seq_len*seq_len);
        
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,CUBLAS_OP_N,
            seq_len,seq_len,dimension,
            &alpha,
            batch_K,dmodel,dimension,
            batch_Q,dmodel,dimension,
            &beta,
            batch_S,seq_len,(seq_len*seq_len),
            heads
        );
    }
    
    if(mask) padding_mask_kernel<<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
    softmax_forward(S);
    
    alpha = 1.0f;
    for(int i = 0; i < batch; i++) 
    {
        float* batch_S=S.get_data()+(i*heads*seq_len*seq_len),*batch_V = V.get_data()+(i*seq_len*dmodel),*batch_Z= Z.get_data()+(i*seq_len*dmodel);
        
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,CUBLAS_OP_N,
            dimension,seq_len,seq_len,
            &alpha,
            batch_V,dmodel,dimension,
            batch_S,seq_len,(seq_len*seq_len),
            &beta,
            batch_Z,dmodel,dimension,
            heads
        );
    }
}

__global__ void softmax_attn_backward_kernel(float* dS,const float* S,int seq_len)
{
    int r=blockIdx.x,t=threadIdx.x,s=blockDim.x;
    
    float local_dot=0.0f;
    for(int i=t;i<seq_len;i+=s)
    {
        int idx=r*seq_len+i;
        local_dot+=dS[idx]*S[idx];
    }
    
    __shared__ float s_dot[256];
    s_dot[t]=local_dot;
    __syncthreads();
    
    for(int s_step=128;s_step>0;s_step>>=1){if(t<s_step) s_dot[t]+=s_dot[t+s_step];__syncthreads();}
    
    __shared__ float r_dot;
    if(t==0) r_dot=s_dot[0];
    __syncthreads();
    
    for(int i=t;i<seq_len;i+=s)
    {
        int idx=r*seq_len+i;
        dS[idx]=S[idx]*(dS[idx]-r_dot);
    }
}

void attention_backward(Tensor& dAttn,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads)
{
    int batch_seq=Q.rows(),d_model=Q.cols();
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    int dimension=d_model/heads;
    
    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=0.0f;
    
    Tensor dS({batch,heads,seq_len,seq_len});
    
    for(int i=0;i<batch;i++)
    {
        float* batch_S=S.get_data()+(i*heads*seq_len*seq_len),*batch_dAttn=dAttn.get_data()+(i*seq_len*d_model),*batch_dV=dV.get_data()+(i*seq_len*d_model);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&alpha,batch_dAttn,d_model,dimension,batch_S,seq_len,(seq_len*seq_len),&beta,batch_dV,d_model,dimension,heads);
    }
    
    for(int i=0;i<batch;i++)
    {
        float* batch_dAttn=dAttn.get_data()+(i*seq_len*d_model),*batch_V=V.get_data()+(i*seq_len*d_model),*batch_dS=dS.get_data()+(i*heads*seq_len*seq_len);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,batch_V,d_model,dimension,batch_dAttn,d_model,dimension,&beta,batch_dS,seq_len,(seq_len*seq_len),heads);
    }
    
    int total=batch*heads*seq_len;
    softmax_attn_backward_kernel<<<total,256>>>(dS.get_data(),S.get_data(),seq_len);
    
    float scale=1.0f/sqrtf((float)dimension);
    for(int i=0;i<batch;i++)
    {
        float* batch_dS=dS.get_data()+(i*heads*seq_len*seq_len),*batch_K=K.get_data()+(i*seq_len*d_model),*batch_dQ=dQ.get_data()+(i*seq_len*d_model);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&scale,batch_K,d_model,dimension,batch_dS,seq_len,(seq_len*seq_len),&beta,batch_dQ,d_model,dimension,heads);
    }
    
    for(int i=0;i<batch;i++)
    {
        float* batch_dS=dS.get_data()+(i*heads*seq_len*seq_len),*batch_Q=Q.get_data()+(i*seq_len*d_model),*batch_dK=dK.get_data()+(i*seq_len*d_model);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&scale,batch_Q,d_model,dimension,batch_dS,seq_len,(seq_len*seq_len),&beta,batch_dK,d_model,dimension,heads);
    }
}

__global__ void masked_attention_forward_kernel(float* S,int seq_len)
{
    int row=blockIdx.x,seq_row=row%seq_len,t=threadIdx.x,s=blockDim.x;
    for(int i=t;i<seq_len;i+=s) if(i>seq_row) S[row*seq_len+i]=-1e9f;
}

__global__ void masked_attention_backward_kernel(float* dS,int seq_len)
{
    int row=blockIdx.x,seq_row=row%seq_len,t=threadIdx.x,s=blockDim.x;
    for(int i=t;i<seq_len;i+=s) if(i>seq_row) dS[row*seq_len+i]=0.0f;
}

void masked_attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Out,int heads,Tensor& S,const Tensor* mask)
{
    int batch_seq=Q.rows(),dmodel=Q.cols();
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    int dimension=dmodel/heads,total=batch*heads*seq_len;

    S=Tensor::zeros({batch,heads,seq_len,seq_len});

    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f/sqrtf((float)dimension),beta=0.0f;

    for(int i=0;i<batch;i++)
    {
        float* bQ=Q.get_data()+(i*seq_len*dmodel),*bK=K.get_data()+(i*seq_len*dmodel),*bS=S.get_data()+(i*heads*seq_len*seq_len);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,bK,dmodel,dimension,bQ,dmodel,dimension,&beta,bS,seq_len,(seq_len*seq_len),heads);
    }

    masked_attention_forward_kernel<<<total,256>>>(S.get_data(),seq_len);
    if(mask) padding_mask_kernel<<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
    softmax_forward(S);
    alpha=1.0f;

    for(int i=0;i<batch;i++)
    {
        float* bS=S.get_data()+(i*heads*seq_len*seq_len),*bV=V.get_data()+(i*seq_len*dmodel),*bOut=Out.get_data()+(i*seq_len*dmodel);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&alpha,bV,dmodel,dimension,bS,seq_len,(seq_len*seq_len),&beta,bOut,dmodel,dimension,heads);
    }
}

void masked_attention_backward(Tensor& dAttn,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads)
{
    int batch_seq=Q.rows(),dmodel=Q.cols();
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    int dimension=dmodel/heads,total=batch*heads*seq_len;

    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=0.0f,scale=1.0f/sqrtf((float)dimension);

    Tensor dS({batch,heads,seq_len,seq_len});

    for(int i=0;i<batch;i++)
    {
        float* bS=S.get_data()+(i*heads*seq_len*seq_len),*bdAttn=dAttn.get_data()+(i*seq_len*dmodel),*bdV=dV.get_data()+(i*seq_len*dmodel);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&alpha,bdAttn,dmodel,dimension,bS,seq_len,(seq_len*seq_len),&beta,bdV,dmodel,dimension,heads);
    }

    for(int i=0;i<batch;i++)
    {
        float* bdAttn=dAttn.get_data()+(i*seq_len*dmodel),*bV=V.get_data()+(i*seq_len*dmodel),*bdS=dS.get_data()+(i*heads*seq_len*seq_len);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,bV,dmodel,dimension,bdAttn,dmodel,dimension,&beta,bdS,seq_len,(seq_len*seq_len),heads);
    }

    softmax_attn_backward_kernel<<<total,256>>>(dS.get_data(),S.get_data(),seq_len);
    masked_attention_backward_kernel<<<total,256>>>(dS.get_data(),seq_len);

    for(int i=0;i<batch;i++)
    {
        float* bdS=dS.get_data()+(i*heads*seq_len*seq_len),*bK=K.get_data()+(i*seq_len*dmodel),*bdQ=dQ.get_data()+(i*seq_len*dmodel);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&scale,bK,dmodel,dimension,bdS,seq_len,(seq_len*seq_len),&beta,bdQ,dmodel,dimension,heads);
    }
    
    for(int i=0;i<batch;i++)
    {
        float* bdS=dS.get_data()+(i*heads*seq_len*seq_len),*bQ=Q.get_data()+(i*seq_len*dmodel),*bdK=dK.get_data()+(i*seq_len*dmodel);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&scale,bQ,dmodel,dimension,bdS,seq_len,(seq_len*seq_len),&beta,bdK,dmodel,dimension,heads);
    }
}
