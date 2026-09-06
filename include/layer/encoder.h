#pragma once
#include "layer.h"
#include "linear.h"
#include "dropout.h"
#include "../core/tensor.h"

class Encoder:public Layer
{
    private:
        int dmodel,heads,dimension;
        Linear Wq,Wk,Wv,Wo,FFN1,FFN2;
        Dropout attn_dropout,ffn_dropout;
        Tensor g1,b1,g2,b2;
        Tensor dg1,db1,dg2,db2;
        Tensor cached_X,cached_attn,cached_Q,cached_K,cached_V,cached_temp,cached_relu;

    public:
        Encoder(int dmodel,int heads,int dff,int layers,float dr=0.1f);

        Tensor forward(Tensor const& X) override;
        Tensor backward(Tensor const& dY) override;

        std::vector<Tensor*> get_weights() override;
        std::vector<Tensor*> get_grads() override;

        void train() override;
        void eval() override;
};
