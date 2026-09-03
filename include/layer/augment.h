#pragma once
#include "layer.h"
#include "../core/augment_ops.h"

class Augment:public Layer
{
    private:
        bool is_training,flip;
        int max,hole,b_;
        void* pgpu;

    public:
        Augment(bool flip=true,int max=4,int hole=8);
        ~Augment() override;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;

        void eval(){is_training=false;}
        void train(){is_training=true;}
};