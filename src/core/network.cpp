#include "../../include/core/network.h"

void Network::add_layer(Layer* layer){this->layers.push_back(layer);}

std::vector<Layer*> Network::get_layers(){return layers;}

void Network::compile(Optimizer* optimizer,Loss* loss){this->optimizer=optimizer;this->loss=loss;}

Tensor Network::forward(Tensor X){for(auto layer:layers)X=layer->forward(X);return X;}

void Network::backward(Tensor dY){for(int i=layers.size()-1;i>=0;i--)dY=layers[i]->backward(dY);}

float Network::train_step(const Tensor& X,const Tensor& Y)
{
    Tensor pred=this->forward(X);
    float loss=this->loss->calculate_loss(pred,Y);
    Tensor dY=this->loss->backward_loss(pred, Y);
    this->backward(dY);
    this->optimizer->step();
    return loss;
    return loss;
}

Network::~Network(){for(auto layer:layers)delete layer;delete optimizer;delete loss;}