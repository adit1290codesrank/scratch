#include "../../../include/core/optimizers_ops.h"

__global__ void adam_kernel(float* W,float* dW, float* m,float *v,float lr,float b1,float b2,float b1t,float b2t,float e,float wd,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size)
    {
        m[index]=b1*m[index]+(1.0-b1)*dW[index];
        v[index]=b2*v[index]+(1.0-b2)*dW[index]*dW[index];

        float m_=m[index]/(1.0-b1t),v_=v[index]/(1.0-b2t);
        
        W[index]-=lr*(m_/(sqrtf(v_)+e) + wd*W[index]);
    }
}

void adam(Tensor *W,Tensor *dW,Tensor& m,Tensor& v,float lr,float b1,float b2,float b1t,float b2t,float e,float wd)
{
    size_t size=W->total_elements();
    int threads=256;
    int blocks=(size+threads-1)/threads;
    adam_kernel<<<blocks,threads>>>(W->get_data(),dW->get_data(),m.get_data(),v.get_data(),lr,b1,b2,b1t,b2t,e,wd,size);
}