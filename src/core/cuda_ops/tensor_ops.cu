#include "../../../include/core/tensor_ops.h"
#include "../../../include/core/context.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include "../../../include/core/dtype_ops.h"

static cudaDataType_t cuda_dtype(DType dt){return dt==DType::F32?CUDA_R_32F:CUDA_R_16BF;}

Tensor multiply(const Tensor& a,bool transA,const Tensor& b,bool transB)
{
    int m=transA?a.cols():a.rows();
    int k1=transA?a.rows():a.cols();
    int k2=transB?b.cols():b.rows();
    int n=transB?b.rows():b.cols();
    if(k1!=k2) throw std::invalid_argument("Inner dimensions must match");

    Tensor c({m,n},a.dtype());
    cublasHandle_t handle = Context::get_instance().get_cublas_handle();

    cublasOperation_t opA=transA?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t opB=transB?CUBLAS_OP_T:CUBLAS_OP_N;

    float alpha=1.0f,beta = 0.0f;
    cudaDataType_t dt=cuda_dtype(a.dtype());
    cublasGemmEx(
        handle,
        opB,
        opA,
        n, m, k1,
        &alpha,
        b.get_data(),dt,b.cols(),
        a.get_data(),dt,a.cols(),
        &beta,
        c.get_data(),dt,c.cols(),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    return c;
}

// Tensor multiply(const Tensor& a,bool transA,const Tensor& b,bool transB)
// {
//     int m=transA?a.cols():a.rows();
//     int k1=transA?a.rows():a.cols();
//     int k2=transB?b.cols():b.rows();
//     int n=transB?b.rows():b.cols();
//     if(k1!=k2) throw std::invalid_argument("Inner dimensions must match");
//
//     Tensor c({m,n});
//     cublasHandle_t handle = Context::get_instance().get_cublas_handle();
//
//     cublasOperation_t opA=transA?CUBLAS_OP_T:CUBLAS_OP_N;
//     cublasOperation_t opB=transB?CUBLAS_OP_T:CUBLAS_OP_N;
//
//     float alpha=1.0f,beta = 0.0f;
//     cublasSgemm(
//         handle,
//         opB,
//         opA,
//         n, m, k1,
//         &alpha,
//         b.get_data(),b.cols(),
//         a.get_data(),a.cols(),
//         &beta,
//         c.get_data(),c.cols()
//     );
//     return c;
// }

template<typename T>
__global__ void add_kernel(const T* a,const T* b,T* c,int total)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<total) c[index]=from_float<T>(to_float(a[index])+to_float(b[index]));
}

Tensor add(const Tensor& a,const Tensor& b)
{
    if(a.shape!=b.shape) throw std::invalid_argument("Tensor shapes must match");

    int total=(int)a.total_elements();
    Tensor c(a.shape,a.dtype());
    int threads=256;
    int blocks=(total+threads-1)/threads;
    if(a.dtype()==DType::F32) add_kernel<float><<<blocks,threads>>>(a.get_data(),b.get_data(),c.get_data(),total);
    else add_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)a.get_data(),(__nv_bfloat16*)b.get_data(),(__nv_bfloat16*)c.get_data(),total);
    return c;
}

// Tensor add(const Tensor& a,const Tensor& b)
// {
//     if(a.shape!=b.shape) throw std::invalid_argument("Tensor shapes must match");
//
//     size_t total=a.total_elements();
//
//     Tensor c(a.shape);
//     cublasHandle_t handle = Context::get_instance().get_cublas_handle();
//
//     float alpha=1.0f,beta = 1.0f;
//     cublasSgeam(
//         handle,
//         CUBLAS_OP_N,
//         CUBLAS_OP_N,
//         total,1,
//         &alpha,
//         a.get_data(),total,
//         &beta,
//         b.get_data(),total,
//         c.get_data(),total
//     );
//     return c;
// }

template<typename T>
__global__ void add_bias_kernel(T *Y,const T *b,int n,int m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int total=n*m;
    if(index<total) Y[index]=from_float<T>(to_float(Y[index])+to_float(b[index%m]));
}

// __global__ void add_bias_kernel(float *Y,const float *b,int n,int m)
// {
//     int index=blockIdx.x*blockDim.x+threadIdx.x;
//     int total=n*m;
//     if(index<total) Y[index]+=b[index%m];
// }

void add_bias(Tensor& Y,const Tensor& b)
{
    int n=Y.rows(),m=Y.cols();
    int total=n*m;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    if(Y.dtype()==DType::F32) add_bias_kernel<float><<<blocks,threads>>>(Y.get_data(),b.get_data(),n,m);
    else add_bias_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)Y.get_data(),(__nv_bfloat16*)b.get_data(),n,m);
}

// void add_bias(Tensor& Y,const Tensor& b)
// {
//     int n=Y.rows(),m=Y.cols();
//     int total=n*m;
//     int threads=256;
//     int blocks=(total+threads-1)/threads;
//     add_bias_kernel<<<blocks,threads>>>(Y.get_data(),b.get_data(),n,m);
// }

template<typename T>
__global__ void sum_rows_kernel(const T *dY,T *db,int n,int m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<m)
    {
        float sum=0.0f;                              
        for(int i=0;i<n;i++) sum+=to_float(dY[i*m+index]);
        db[index]=from_float<T>(sum);
    }
}

// __global__ void sum_rows_kernel(const float *dY,float *db,int n,int m)
// {
//     int index=blockIdx.x*blockDim.x+threadIdx.x;
//     if(index<m)
//     {
//         float sum=0.0f;
//         for(int i=0;i<n;i++) sum+=dY[i*m+index];
//         db[index]=sum;
//     }
// }

Tensor sum_rows(const Tensor& dY)
{
    int n=dY.rows(),m=dY.cols();
    Tensor db({1,m},dY.dtype());
    int threads=256;
    int blocks=(m+threads-1)/threads;
    if(dY.dtype()==DType::F32) sum_rows_kernel<float><<<blocks,threads>>>(dY.get_data(),db.get_data(),n,m);
    else sum_rows_kernel<__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)dY.get_data(),(__nv_bfloat16*)db.get_data(),n,m);
    return db;
}

// Tensor sum_rows(const Tensor& dY)
// {
//     int n=dY.rows(),m=dY.cols();
//     Tensor db({1,m});
//     int threads=256;
//     int blocks=(m+threads-1)/threads;
//     sum_rows_kernel<<<blocks,threads>>>(dY.get_data(),db.get_data(),n,m);
//     return db;
// }

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

template<typename T>
__global__ void padding_mask_kernel(T* S,const float* mask,int heads,int seq_len)
{
    int row=blockIdx.x,b=row/(heads*seq_len),t=threadIdx.x,s=blockDim.x;
    for(int i=t;i<seq_len;i+=s) if(mask[b*seq_len+i]==0.0f) S[row*seq_len+i]=from_float<T>(-1e9f);
}

// __global__ void padding_mask_kernel(float* S,const float* mask,int heads,int seq_len)
// {
//     int row=blockIdx.x,b=row/(heads*seq_len),t=threadIdx.x,s=blockDim.x;
//     for(int i=t;i<seq_len;i+=s) if(mask[b*seq_len+i]==0.0f) S[row*seq_len+i]=-1e9f;
// }

void attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Z,int heads,Tensor& S,const Tensor* mask)
{
    int batch_seq=Q.rows(),dmodel=Q.shape.back();
    
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
    

    //if(mask) padding_mask_kernel<<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
    if(mask)
    {
        if(S.dtype()==DType::F32) padding_mask_kernel<float><<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
        else padding_mask_kernel<__nv_bfloat16><<<batch*heads*seq_len,256>>>((__nv_bfloat16*)S.get_data(),mask->get_data(),heads,seq_len);
    }

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

template<typename T>
__global__ void softmax_attn_backward_kernel(T* dS,const T* S,int seq_len)
{
    int r=blockIdx.x,t=threadIdx.x,s=blockDim.x;

    float local_dot=0.0f;
    for(int i=t;i<seq_len;i+=s)
    {
        int idx=r*seq_len+i;
        local_dot+=to_float(dS[idx])*to_float(S[idx]);
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
        dS[idx]=from_float<T>(to_float(S[idx])*(to_float(dS[idx])-r_dot));
    }
}

void attention_backward(Tensor& dZ,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads)
{
    int batch_seq=Q.rows(),d_model=Q.shape.back();
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    int dimension=d_model/heads;
    
    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=0.0f;
    
    Tensor dS({batch,heads,seq_len,seq_len});
    
    for(int i=0;i<batch;i++)
    {
        float* batch_S=S.get_data()+(i*heads*seq_len*seq_len),*batch_dZ=dZ.get_data()+(i*seq_len*d_model),*batch_dV=dV.get_data()+(i*seq_len*d_model);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&alpha,batch_dZ,d_model,dimension,batch_S,seq_len,(seq_len*seq_len),&beta,batch_dV,d_model,dimension,heads);
    }
    
    for(int i=0;i<batch;i++)
    {
        float* batch_dZ=dZ.get_data()+(i*seq_len*d_model),*batch_V=V.get_data()+(i*seq_len*d_model),*batch_dS=dS.get_data()+(i*heads*seq_len*seq_len);
        cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,batch_V,d_model,dimension,batch_dZ,d_model,dimension,&beta,batch_dS,seq_len,(seq_len*seq_len),heads);
    }
    
    int total=batch*heads*seq_len;
    if(dS.dtype()==DType::F32) softmax_attn_backward_kernel<float><<<total,256>>>(dS.get_data(),S.get_data(),seq_len);
    else softmax_attn_backward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)dS.get_data(),(__nv_bfloat16*)S.get_data(),seq_len);

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

template<typename T>
__global__ void masked_attention_forward_kernel(T* S,int seq_len)
{
    int row=blockIdx.x,seq_row=row%seq_len,t=threadIdx.x,s=blockDim.x;
    for(int i=t;i<seq_len;i+=s) if(i>seq_row) S[row*seq_len+i]=from_float<T>(-1e9f);
}

template<typename T>
__global__ void masked_attention_backward_kernel(T* dS,int seq_len)
{
    int row=blockIdx.x,seq_row=row%seq_len,t=threadIdx.x,s=blockDim.x;
    for(int i=t;i<seq_len;i+=s) if(i>seq_row) dS[row*seq_len+i]=from_float<T>(0.0f);
}

// __global__ void masked_attention_forward_kernel(float* S,int seq_len)
// {
//     int row=blockIdx.x,seq_row=row%seq_len,t=threadIdx.x,s=blockDim.x;
//     for(int i=t;i<seq_len;i+=s) if(i>seq_row) S[row*seq_len+i]=-1e9f;
// }

// __global__ void masked_attention_backward_kernel(float* dS,int seq_len)
// {
//     int row=blockIdx.x,seq_row=row%seq_len,t=threadIdx.x,s=blockDim.x;
//     for(int i=t;i<seq_len;i+=s) if(i>seq_row) dS[row*seq_len+i]=0.0f;
// }

void masked_attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Z,int heads,Tensor& S,const Tensor* mask)
{
    int batch_seq=Q.rows(),dmodel=Q.shape.back();
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    int dimension=dmodel/heads,total=batch*heads*seq_len;

    S=Tensor::zeros({batch,heads,seq_len,seq_len},Q.dtype());

    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f/sqrtf((float)dimension),beta=0.0f;
    cudaDataType_t dt=cuda_dtype(Q.dtype());

    for(int i=0;i<batch;i++)
    {
        char* bQ=(char*)Q.get_data()+(size_t)(i*seq_len*dmodel)*Q.elem_size();
        char* bK=(char*)K.get_data()+(size_t)(i*seq_len*dmodel)*K.elem_size();
        char* bS=(char*)S.get_data()+(size_t)(i*heads*seq_len*seq_len)*S.elem_size();
        cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,
            bK,dt,dmodel,dimension,
            bQ,dt,dmodel,dimension,
            &beta,
            bS,dt,seq_len,(seq_len*seq_len),
            heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }

    if(S.dtype()==DType::F32) masked_attention_forward_kernel<float><<<total,256>>>(S.get_data(),seq_len);
    else masked_attention_forward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)S.get_data(),seq_len);
    if(mask)
    {
        if(S.dtype()==DType::F32) padding_mask_kernel<float><<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
        else padding_mask_kernel<__nv_bfloat16><<<batch*heads*seq_len,256>>>((__nv_bfloat16*)S.get_data(),mask->get_data(),heads,seq_len);
    }

    softmax_forward(S);
    alpha=1.0f;

    for(int i=0;i<batch;i++)
    {
        char* bS=(char*)S.get_data()+(size_t)(i*heads*seq_len*seq_len)*S.elem_size();
        char* bV=(char*)V.get_data()+(size_t)(i*seq_len*dmodel)*V.elem_size();
        char* bZ=(char*)Z.get_data()+(size_t)(i*seq_len*dmodel)*Z.elem_size();
        cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&alpha,
            bV,dt,dmodel,dimension,
            bS,dt,seq_len,(seq_len*seq_len),
            &beta,
            bZ,dt,dmodel,dimension,
            heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }
}

// void masked_attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Z,int heads,Tensor& S,const Tensor* mask)
// {
//     int batch_seq=Q.rows(),dmodel=Q.shape.back();
//     int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
//     int dimension=dmodel/heads,total=batch*heads*seq_len;
//
//     S=Tensor::zeros({batch,heads,seq_len,seq_len},Q.dtype());
//
//     cublasHandle_t handle=Context::get_instance().get_cublas_handle();
//     float alpha=1.0f/sqrtf((float)dimension),beta=0.0f;
//     cudaDataType_t dt=cuda_dtype(Q.dtype());
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bQ=Q.get_data()+(i*seq_len*dmodel),*bK=K.get_data()+(i*seq_len*dmodel),*bS=S.get_data()+(i*heads*seq_len*seq_len);
//         cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,
//             bK,dt,dmodel,dimension,
//             bQ,dt,dmodel,dimension,
//             &beta,
//             bS,dt,seq_len,(seq_len*seq_len),
//             heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
//     }
//
//     if(S.dtype()==DType::F32) masked_attention_forward_kernel<float><<<total,256>>>(S.get_data(),seq_len);
//     else masked_attention_forward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)S.get_data(),seq_len);
//     if(mask)
//     {
//         if(S.dtype()==DType::F32) padding_mask_kernel<float><<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
//         else padding_mask_kernel<__nv_bfloat16><<<batch*heads*seq_len,256>>>((__nv_bfloat16*)S.get_data(),mask->get_data(),heads,seq_len);
//     }
//
//     softmax_forward(S);
//     alpha=1.0f;
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bS=S.get_data()+(i*heads*seq_len*seq_len),*bV=V.get_data()+(i*seq_len*dmodel),*bZ=Z.get_data()+(i*seq_len*dmodel);
//         cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&alpha,
//             bV,dt,dmodel,dimension,
//             bS,dt,seq_len,(seq_len*seq_len),
//             &beta,
//             bZ,dt,dmodel,dimension,
//             heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
//     }
// }

// void masked_attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Z,int heads,Tensor& S,const Tensor* mask)
// {
//     int batch_seq=Q.rows(),dmodel=Q.shape.back();
//     int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
//     int dimension=dmodel/heads,total=batch*heads*seq_len;
//
//     S=Tensor::zeros({batch,heads,seq_len,seq_len});
//
//     cublasHandle_t handle=Context::get_instance().get_cublas_handle();
//     float alpha=1.0f/sqrtf((float)dimension),beta=0.0f;
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bQ=Q.get_data()+(i*seq_len*dmodel),*bK=K.get_data()+(i*seq_len*dmodel),*bS=S.get_data()+(i*heads*seq_len*seq_len);
//         cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,bK,dmodel,dimension,bQ,dmodel,dimension,&beta,bS,seq_len,(seq_len*seq_len),heads);
//     }
//
//     if(S.dtype()==DType::F32) masked_attention_forward_kernel<float><<<total,256>>>(S.get_data(),seq_len);
//     else masked_attention_forward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)S.get_data(),seq_len);
//     if(mask)
//     {
//         if(S.dtype()==DType::F32) padding_mask_kernel<float><<<batch*heads*seq_len,256>>>(S.get_data(),mask->get_data(),heads,seq_len);
//         else padding_mask_kernel<__nv_bfloat16><<<batch*heads*seq_len,256>>>((__nv_bfloat16*)S.get_data(),mask->get_data(),heads,seq_len);
//     }
//
//     softmax_forward(S);
//     alpha=1.0f;
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bS=S.get_data()+(i*heads*seq_len*seq_len),*bV=V.get_data()+(i*seq_len*dmodel),*bZ=Z.get_data()+(i*seq_len*dmodel);
//         cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&alpha,bV,dmodel,dimension,bS,seq_len,(seq_len*seq_len),&beta,bZ,dmodel,dimension,heads);
//     }
// }

void masked_attention_backward(Tensor& dZ,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads)
{
    int batch_seq=Q.rows(),dmodel=Q.shape.back();
    int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
    int dimension=dmodel/heads,total=batch*heads*seq_len;

    cublasHandle_t handle=Context::get_instance().get_cublas_handle();
    float alpha=1.0f,beta=0.0f,scale=1.0f/sqrtf((float)dimension);
    cudaDataType_t dt=cuda_dtype(Q.dtype());

    Tensor dS({batch,heads,seq_len,seq_len},Q.dtype());

    for(int i=0;i<batch;i++)
    {
        char* bS=(char*)S.get_data()+(size_t)(i*heads*seq_len*seq_len)*S.elem_size();
        char* bdZ=(char*)dZ.get_data()+(size_t)(i*seq_len*dmodel)*dZ.elem_size();
        char* bdV=(char*)dV.get_data()+(size_t)(i*seq_len*dmodel)*dV.elem_size();
        cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&alpha,
            bdZ,dt,dmodel,dimension,
            bS,dt,seq_len,(seq_len*seq_len),
            &beta,
            bdV,dt,dmodel,dimension,
            heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }

    for(int i=0;i<batch;i++)
    {
        char* bdZ=(char*)dZ.get_data()+(size_t)(i*seq_len*dmodel)*dZ.elem_size();
        char* bV=(char*)V.get_data()+(size_t)(i*seq_len*dmodel)*V.elem_size();
        char* bdS=(char*)dS.get_data()+(size_t)(i*heads*seq_len*seq_len)*dS.elem_size();
        cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,
            bV,dt,dmodel,dimension,
            bdZ,dt,dmodel,dimension,
            &beta,
            bdS,dt,seq_len,(seq_len*seq_len),
            heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }

    if(dS.dtype()==DType::F32) softmax_attn_backward_kernel<float><<<total,256>>>(dS.get_data(),S.get_data(),seq_len);
    else softmax_attn_backward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)dS.get_data(),(__nv_bfloat16*)S.get_data(),seq_len);
    if(dS.dtype()==DType::F32) masked_attention_backward_kernel<float><<<total,256>>>(dS.get_data(),seq_len);
    else masked_attention_backward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)dS.get_data(),seq_len);

    for(int i=0;i<batch;i++)
    {
        char* bdS=(char*)dS.get_data()+(size_t)(i*heads*seq_len*seq_len)*dS.elem_size();
        char* bK=(char*)K.get_data()+(size_t)(i*seq_len*dmodel)*K.elem_size();
        char* bdQ=(char*)dQ.get_data()+(size_t)(i*seq_len*dmodel)*dQ.elem_size();
        cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&scale,
            bK,dt,dmodel,dimension,
            bdS,dt,seq_len,(seq_len*seq_len),
            &beta,
            bdQ,dt,dmodel,dimension,
            heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }

    for(int i=0;i<batch;i++)
    {
        char* bdS=(char*)dS.get_data()+(size_t)(i*heads*seq_len*seq_len)*dS.elem_size();
        char* bQ=(char*)Q.get_data()+(size_t)(i*seq_len*dmodel)*Q.elem_size();
        char* bdK=(char*)dK.get_data()+(size_t)(i*seq_len*dmodel)*dK.elem_size();
        cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&scale,
            bQ,dt,dmodel,dimension,
            bdS,dt,seq_len,(seq_len*seq_len),
            &beta,
            bdK,dt,dmodel,dimension,
            heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }
}

// void masked_attention_backward(Tensor& dZ,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads)
// {
//     int batch_seq=Q.rows(),dmodel=Q.shape.back();
//     int batch=(Q.shape.size()>2)?Q.shape[0]:1,seq_len=(Q.shape.size()>2)?Q.shape[1]:batch_seq;
//     int dimension=dmodel/heads,total=batch*heads*seq_len;
//
//     cublasHandle_t handle=Context::get_instance().get_cublas_handle();
//     float alpha=1.0f,beta=0.0f,scale=1.0f/sqrtf((float)dimension);
//
//     Tensor dS({batch,heads,seq_len,seq_len});
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bS=S.get_data()+(i*heads*seq_len*seq_len),*bdZ=dZ.get_data()+(i*seq_len*dmodel),*bdV=dV.get_data()+(i*seq_len*dmodel);
//         cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&alpha,bdZ,dmodel,dimension,bS,seq_len,(seq_len*seq_len),&beta,bdV,dmodel,dimension,heads);
//     }
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bdZ=dZ.get_data()+(i*seq_len*dmodel),*bV=V.get_data()+(i*seq_len*dmodel),*bdS=dS.get_data()+(i*heads*seq_len*seq_len);
//         cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,seq_len,seq_len,dimension,&alpha,bV,dmodel,dimension,bdZ,dmodel,dimension,&beta,bdS,seq_len,(seq_len*seq_len),heads);
//     }
//
//     if(dS.dtype()==DType::F32) softmax_attn_backward_kernel<float><<<total,256>>>(dS.get_data(),S.get_data(),seq_len);
//     else softmax_attn_backward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)dS.get_data(),(__nv_bfloat16*)S.get_data(),seq_len);
//     if(dS.dtype()==DType::F32) masked_attention_backward_kernel<float><<<total,256>>>(dS.get_data(),seq_len);
//     else masked_attention_backward_kernel<__nv_bfloat16><<<total,256>>>((__nv_bfloat16*)dS.get_data(),seq_len);
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bdS=dS.get_data()+(i*heads*seq_len*seq_len),*bK=K.get_data()+(i*seq_len*dmodel),*bdQ=dQ.get_data()+(i*seq_len*dmodel);
//         cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,dimension,seq_len,seq_len,&scale,bK,dmodel,dimension,bdS,seq_len,(seq_len*seq_len),&beta,bdQ,dmodel,dimension,heads);
//     }
//
//     for(int i=0;i<batch;i++)
//     {
//         float* bdS=dS.get_data()+(i*heads*seq_len*seq_len),*bQ=Q.get_data()+(i*seq_len*dmodel),*bdK=dK.get_data()+(i*seq_len*dmodel);
//         cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,dimension,seq_len,seq_len,&scale,bQ,dmodel,dimension,bdS,seq_len,(seq_len*seq_len),&beta,bdK,dmodel,dimension,heads);
//     }
// }

template<typename T>
__global__ void embedding_forward_kernel(const float* X,const T* W_tok,const T* W_pos,T* Y,int seq_len,int dmodel,int total)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x,s=blockDim.x*gridDim.x;
    for(int i=idx;i<total;i+=s)
    {
        int d=i%dmodel,seq=(i/dmodel)%seq_len,tok=(int)X[i/dmodel];
        Y[i]=from_float<T>(to_float(W_tok[tok*dmodel+d])+to_float(W_pos[seq*dmodel+d]));
    }
}

// template<typename T>
// __global__ void embedding_forward_kernel(const float* X,const float* W_tok,const float* W_pos,T* Y,int seq_len,int dmodel,int total)
// {
//     int idx=blockIdx.x*blockDim.x+threadIdx.x,s=blockDim.x*gridDim.x;
//     for(int i=idx;i<total;i+=s)
//     {
//         int d=i%dmodel,seq=(i/dmodel)%seq_len,tok=(int)X[i/dmodel];
//         Y[i]=from_float<T>(W_tok[tok*dmodel+d]+W_pos[seq*dmodel+d]);
//     }
// }

// __global__ void embedding_forward_kernel(const float* X,const float* W_tok,const float* W_pos,float* Y,int seq_len,int dmodel,int total)
// {
//     int idx=blockIdx.x*blockDim.x+threadIdx.x,s=blockDim.x*gridDim.x;
//     for(int i=idx;i<total;i+=s)
//     {
//         int d=i%dmodel,seq=(i/dmodel)%seq_len,tok=(int)X[i/dmodel];
//         Y[i]=W_tok[tok*dmodel+d]+W_pos[seq*dmodel+d];
//     }
// }

template<typename T>
__global__ void embedding_backward_kernel(const float* X,const T* dY,float* dW_tok,float* dW_pos,int seq_len,int dmodel,int total)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x,s=blockDim.x*gridDim.x;
    for(int i=idx;i<total;i+=s)
    {
        int d=i%dmodel,seq=(i/dmodel)%seq_len,tok=(int)X[i/dmodel];
        float g=to_float(dY[i]);
        atomicAdd(&dW_tok[tok*dmodel+d],g);
        atomicAdd(&dW_pos[seq*dmodel+d],g);
    }
}

// __global__ void embedding_backward_kernel(const float* X,const float* dY,float* dW_tok,float* dW_pos,int seq_len,int dmodel,int total)
// {
//     int idx=blockIdx.x*blockDim.x+threadIdx.x,s=blockDim.x*gridDim.x;
//     for(int i=idx;i<total;i+=s)
//     {
//         int d=i%dmodel,seq=(i/dmodel)%seq_len,tok=(int)X[i/dmodel];
//         atomicAdd(&dW_tok[tok*dmodel+d],dY[i]);
//         atomicAdd(&dW_pos[seq*dmodel+d],dY[i]);
//     }
// }

void embedding_forward(const Tensor& X,const Tensor& W_tok,const Tensor& W_pos,Tensor& Y,int batch,int seq_len,int dmodel)
{
    int total=batch*seq_len*dmodel,blocks=(total+255)/256;
    if(Y.dtype()==DType::F32)
        embedding_forward_kernel<float><<<blocks,256>>>(X.get_data(),W_tok.get_data(),W_pos.get_data(),Y.get_data(),seq_len,dmodel,total);
    else
        embedding_forward_kernel<__nv_bfloat16><<<blocks,256>>>(X.get_data(),(__nv_bfloat16*)W_tok.get_data(),(__nv_bfloat16*)W_pos.get_data(),(__nv_bfloat16*)Y.get_data(),seq_len,dmodel,total);
}

// void embedding_forward(const Tensor& X,const Tensor& W_tok,const Tensor& W_pos,Tensor& Y,int batch,int seq_len,int dmodel)
// {
//     int total=batch*seq_len*dmodel,blocks=(total+255)/256;
//     if(Y.dtype()==DType::F32) embedding_forward_kernel<float><<<blocks,256>>>(X.get_data(),W_tok.get_data(),W_pos.get_data(),Y.get_data(),seq_len,dmodel,total);
//     else embedding_forward_kernel<__nv_bfloat16><<<blocks,256>>>(X.get_data(),W_tok.get_data(),W_pos.get_data(),(__nv_bfloat16*)Y.get_data(),seq_len,dmodel,total);
// }

// void embedding_forward(const Tensor& X,const Tensor& W_tok,const Tensor& W_pos,Tensor& Y,int batch,int seq_len,int dmodel)
// {
//     int total=batch*seq_len*dmodel,blocks=(total+255)/256;
//     embedding_forward_kernel<<<blocks,256>>>(X.get_data(),W_tok.get_data(),W_pos.get_data(),Y.get_data(),seq_len,dmodel,total);
// }

void embedding_backward(const Tensor& X,const Tensor& dY,Tensor& dW_tok,Tensor& dW_pos,int batch,int seq_len,int dmodel)
{
    int total=batch*seq_len*dmodel,blocks=(total+255)/256;
    if(dY.dtype()==DType::F32) embedding_backward_kernel<float><<<blocks,256>>>(X.get_data(),dY.get_data(),dW_tok.get_data(),dW_pos.get_data(),seq_len,dmodel,total);
    else embedding_backward_kernel<__nv_bfloat16><<<blocks,256>>>(X.get_data(),(__nv_bfloat16*)dY.get_data(),dW_tok.get_data(),dW_pos.get_data(),seq_len,dmodel,total);
}

// void embedding_backward(const Tensor& X,const Tensor& dY,Tensor& dW_tok,Tensor& dW_pos,int batch,int seq_len,int dmodel)
// {
//     int total=batch*seq_len*dmodel,blocks=(total+255)/256;
//     embedding_backward_kernel<<<blocks,256>>>(X.get_data(),dY.get_data(),dW_tok.get_data(),dW_pos.get_data(),seq_len,dmodel,total);
// }

template<typename SrcT,typename DstT>
__global__ void cast_kernel(const SrcT* src,DstT* dst,int total)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<total) dst[i]=from_float<DstT>(to_float(src[i]));
}

void cast_tensor(Tensor& dst,const Tensor& src)
{
    if(dst.total_elements()!=src.total_elements()) throw std::invalid_argument("cast_tensor: element count mismatch");
    int total=(int)src.total_elements();
    int threads=256;
    int blocks=(total+threads-1)/threads;
    if(src.dtype()==DType::F32&&dst.dtype()==DType::F32)
        cast_kernel<float,float><<<blocks,threads>>>(src.get_data(),dst.get_data(),total);
    else if(src.dtype()==DType::F32&&dst.dtype()==DType::BF16)
        cast_kernel<float,__nv_bfloat16><<<blocks,threads>>>(src.get_data(),(__nv_bfloat16*)dst.get_data(),total);
    else if(src.dtype()==DType::BF16&&dst.dtype()==DType::F32)
        cast_kernel<__nv_bfloat16,float><<<blocks,threads>>>((__nv_bfloat16*)src.get_data(),dst.get_data(),total);
    else
        cast_kernel<__nv_bfloat16,__nv_bfloat16><<<blocks,threads>>>((__nv_bfloat16*)src.get_data(),(__nv_bfloat16*)dst.get_data(),total);
}
