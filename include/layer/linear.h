#pragma once
#include "../../include/layer/layer.h"
#include "../../include/core/tensor.h"
#include "../../include/util/init.h"

class Linear:public Layer
{
    private:
        Tensor W;
        Tensor b;
        Tensor cached_X;
        Tensor dW;
        Tensor db;
    
    public:

        Linear(int in,int out,Init init=Init::KAIMING,float custom_std=-1.0f);
        Tensor forward(const Tensor& X,const Tensor* pad_mask=nullptr) override;
        Tensor backward(Tensor const& dY) override;

        std::vector<Tensor*> get_weights() override;
        std::vector<Tensor*> get_grads() override;
};