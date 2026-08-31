#pragma once
#include "layer.h"
#include "../core/tensor.h"
#include "../util/init.h"
#include <vector>

class Conv2D:public Layer
{
    private:
        int cin,cout,k,s,p;

        Tensor W,b,cached_X,cached_Xcol;

        Tensor dW,db;

    public:
        Conv2D(int cin,int cout,int k,int s=1,int p=0,Init init=Init::KAIMING);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;

        std::vector<Tensor*> get_weights() override;
        std::vector<Tensor*> get_grads() override;
};