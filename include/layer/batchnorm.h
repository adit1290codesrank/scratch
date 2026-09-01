#pragma once
#include "../core/tensor.h"
#include "layer.h"
#include <vector>

class BatchNorm:public Layer
{
    private:
        int n;
        float e,m;
        bool is_training;

        Tensor g,b;
        Tensor dg,db;
        Tensor rm,rv;
        Tensor cached_X,cached_X_;
        Tensor cached_bmean,cached_bvar;

    public:
        BatchNorm(int n,float e=1e-5f,float m=0.1f);
        ~BatchNorm()=default;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;

        std::vector<Tensor*> get_weights() override {return {&g,&b};}
        std::vector<Tensor*> get_grads() override {return {&dg,&db};}
        std::vector<Tensor*> get_states() override {return {&rm,&rv};}

        void eval(){is_training=false;}
        void train(){is_training=true;}

};