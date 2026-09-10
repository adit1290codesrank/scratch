#include "../../../include/core/grad_ops.h"
#include "../../../include/core/context.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

void zero_grads(const std::vector<Tensor*>& grads)
{
    for (Tensor* g:grads)
    {
        size_t bytes=g->total_elements()*sizeof(float);
        if (bytes) cudaMemset(g->get_data(),0,bytes);
    }
}

float clip_grad_norm(const std::vector<Tensor*>& grads,float max_norm)
{
    cublasHandle_t h=Context::get_instance().get_cublas_handle();

    double total=0.0;
    for (Tensor* g:grads)
    {
        int n=(int)g->total_elements();
        if (n<=0) continue;
        float norm=0.0f;
        cublasSnrm2(h,n,g->get_data(),1,&norm);
        total+=(double)norm*(double)norm;
    }

    float total_norm=(float)std::sqrt(total);
    if (total_norm>max_norm && total_norm>0.0f)
    {
        float scale=max_norm/(total_norm+1e-6f);
        for (Tensor* g:grads)
        {
            int n=(int)g->total_elements();
            if (n<=0) continue;
            cublasSscal(h,n,&scale,g->get_data(),1);
        }
    }
    return total_norm;
}