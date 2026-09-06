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
        virtual Tensor forward(const Tensor& X,const Tensor* pad_mask=nullptr)=0;
        virtual Tensor backward(const Tensor& dY)=0;
};

class ReLU:public Activation
{
    public:
        Tensor forward(const Tensor& X,const Tensor* pad_mask=nullptr) override;
        Tensor backward(const Tensor& dY) override;
};

class Softmax:public Activation
{
    public:
        Tensor forward(const Tensor& X,const Tensor* pad_mask=nullptr) override;
        Tensor backward(const Tensor& dY) override;
};
