#include "../include/core/tensor.h"
#include "../include/core/optimizer.h"
#include "../include/core/tensor_ops.h"
#include "../include/core/memory.h"
#include <iostream>
#include <vector>
#include <cmath>

class FakeLayer : public Layer
{
    public:
        Tensor W, dW;
        FakeLayer(Tensor w, Tensor g):W(w),dW(g){}
        Tensor forward(const Tensor& X,const Tensor* =nullptr) override { return X; }
        Tensor backward(const Tensor& dY) override { return dY; }
        std::vector<Tensor*> get_weights() override { return {&W}; }
        std::vector<Tensor*> get_grads() override { return {&dW}; }
};

int main(int argc,char** argv)
{
    const int N=16;
    const int STEPS=argc>1?std::atoi(argv[1]):20;

    Tensor w_ref=Tensor::randn({N},0.0f,1.0f);
    Tensor w_bf16({N},DType::BF16);
    cast_tensor(w_bf16,w_ref);

    Tensor g_ref=Tensor::zeros({N});
    Tensor g_bf16=Tensor::zeros({N});

    FakeLayer layer_ref(w_ref,g_ref);
    FakeLayer layer_bf16(w_bf16,g_bf16);

    Adam opt_ref({&layer_ref},0.1f,0.9f,0.999f,1e-8f,0.0f);
    Adam opt_bf16({&layer_bf16},0.1f,0.9f,0.999f,1e-8f,0.0f);

    std::vector<float> grad(N);
    for(int step=0;step<STEPS;step++)
    {
        for(int i=0;i<N;i++) grad[i]=0.05f*((float)((i*7+step*13)%11)-5.0f);
        g_ref.copy_from_host(grad.data());
        opt_ref.step();
        g_bf16.copy_from_host(grad.data());
        opt_bf16.step();
    }

    std::vector<float> final_ref(N),final_bf16(N);
    w_ref.copy_to_host(final_ref.data());
    w_bf16.copy_to_host(final_bf16.data());

    float max_abs=0.0f,max_rel=0.0f;
    for(int i=0;i<N;i++)
    {
        float d=std::fabs(final_ref[i]-final_bf16[i]);
        max_abs=std::max(max_abs,d);
        max_rel=std::max(max_rel,d/(std::fabs(final_ref[i])+1e-6f));
    }

    std::cout<<"after "<<STEPS<<" steps: max abs diff="<<max_abs<<"  max rel diff="<<max_rel<<"\n";
    std::cout<<(max_rel<0.05f?"PASS (within BF16 precision)":"FAIL (diverged beyond BF16 precision)")<<"\n";

    clear_memory_pool();
    return max_rel<0.05f?0:1;
}
