#pragma once
#include "layer.h"
#include "../core/tensor.h"

class Embedding:public Layer
{
    private:
        int vocab_size,dmodel,max_seq;
        Tensor W_token,W_pos;
        Tensor dW_token,dW_pos;
        Tensor cached_X;
        DType dt;

    public:
        Embedding(int vocab_size,int dmodel,int max_seq=1024, DType dt=DType::F32);

        Tensor forward(const Tensor& X,const Tensor* pad_mask=nullptr) override;
        Tensor backward(const Tensor& dY) override;

        std::vector<Tensor*> get_weights() override;
        std::vector<Tensor*> get_grads() override;
};
