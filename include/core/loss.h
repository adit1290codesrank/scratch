#pragma once
#include "tensor.h"

class Loss
{
    public:
        virtual ~Loss()=default;
        virtual float calculate_loss(const Tensor& pred,const Tensor& target)=0;
        virtual Tensor backward_loss(const Tensor& pred,const Tensor& target)=0;
};

class MSELoss:public Loss
{
    public:
        float calculate_loss(const Tensor& pred,const Tensor& target) override;
        Tensor backward_loss(const Tensor& pred,const Tensor& target) override;
};

class CrossEntropyLoss:public Loss
{
    public:
        float calculate_loss(const Tensor& pred,const Tensor& target) override;
        Tensor backward_loss(const Tensor& pred,const Tensor& target) override;
};
