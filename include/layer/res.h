#pragma once
#include "../core/tensor.h"
#include "layer.h"

class Res:public Layer
{
    private:
        std::vector<Layer*> main,skip;

    public:
        Res()=default;
        ~Res() override;

        void add_main(Layer* layer);
        void add_skip(Layer* layer);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;

        std::vector<Tensor*> get_weights() override;
        std::vector<Tensor*> get_grads() override;
        std::vector<Tensor*> get_states() override;
};
