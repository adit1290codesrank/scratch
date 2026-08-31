#pragma once
#include "layer.h"
#include "../core/tensor.h"

class Activation:public Layer
{
    protected:
        Tensor cached_X;
    
    public:
        Activation() : cached_X({0}) {}
        virtual ~Activation() = default;
        virtual Tensor forward(const Tensor& X)=0;
        virtual Tensor backward(const Tensor& dY)=0;
};

class ReLU:public Activation
{
    public:
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;
};

