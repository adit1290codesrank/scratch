#pragma once
#include "tensor.h"
#include "optimizer.h"
#include "../layer/layer.h"
#include "loss.h"
#include <vector>
#include <string>

class Network
{
    private:
        std::vector<Layer*> layers;
        Optimizer* optimizer;
        Loss* loss;

    public:
        Network():optimizer(nullptr),loss(nullptr){}
        ~Network();
        
        void add_layer(Layer* layer);
        std::vector<Layer*> get_layers();
        void compile(Optimizer* optimizer,Loss* loss);

        Tensor forward(Tensor X);
        void backward(Tensor dY);

        std::pair<float,float> train_step(const Tensor& X,const Tensor& Y);

        void save_weights(const std::string& filepath);
        void load_weights(const std::string& filepath);
};