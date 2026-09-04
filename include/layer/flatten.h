#pragma once
#include "layer.h"
#include "../core/tensor.h"
#include <vector>

class Flatten:public Layer
{
    private:
        std::vector<int> cached_shape;

    public:
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;
};
