#pragma once
#include "../core/tensor.h"
#include "layer.h"
class Dropout:public Layer
{
    private:
        float p;
        bool is_training;
        Tensor mask;
    public:
        Dropout(float p=0.3f);
        ~Dropout()=default;
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;
        std::vector<Tensor*> get_weights() override {return {};}
        std::vector<Tensor*> get_grads() override {return {};}
        std::vector<Tensor*> get_states() override {return {};}
        void eval(){is_training=false;}
        void train(){is_training=true;}
};