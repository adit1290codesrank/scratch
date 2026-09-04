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

class LSCrossEntropyLoss:public Loss
{
    private:
        int n;
        float a;

    public:
        LSCrossEntropyLoss(int n,float a=0.1);

        float calculate_loss(const Tensor& pred,const Tensor& target) override;
        Tensor backward_loss(const Tensor& pred,const Tensor& target) override;
};