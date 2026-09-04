#pragma once
#include "../core/tensor.h"
#include <vector>

class Layer
{
    public:
        virtual ~Layer()=default;

        virtual Tensor forward(const Tensor& X)=0;
        virtual Tensor backward(const Tensor& dY)=0;

        virtual std::vector<Tensor*> get_weights() {return {};}
        virtual std::vector<Tensor*> get_grads() {return {};}
        virtual std::vector<Tensor*> get_states() {return {};}

        virtual void eval(){}
        virtual void train(){}
};
