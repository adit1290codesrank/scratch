#pragma once
#include "layer.h"
#include "../core/tensor.h"

class MaxPool:public Layer
{
    private:
        int k,s;
        Tensor cached_X,cached_mask;
    public:
        MaxPool(int k=2,int s=2);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;
};