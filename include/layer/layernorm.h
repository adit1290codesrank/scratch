#pragma once
#include "layer.h"
#include "../core/tensor.h"

class LayerNorm:public Layer
{
    private:
        int dimension;
        Tensor g,b;
        Tensor dg,db;
        Tensor cached_X;

    public:
        LayerNorm(int dimension);

        Tensor forward(const Tensor& X,const Tensor* mask=nullptr) override;
        Tensor backward(const Tensor& dY) override;

        std::vector<Tensor*> get_weights() override;
        std::vector<Tensor*> get_grads() override;
};
