#include "../../include/core/network.h"
#include <string>
#include <fstream>
#include <stdexcept>

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

void Network::save_weights(const std::string& filepath)
{
    std::ofstream file(filepath,std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file for saving weights!");

    for(auto layer:layers)
    {
        for(auto weight:layer->get_weights())
        {
            size_t size=weight->total_elements();
            std::vector<float> data(size);
            weight->copy_to_host(data.data());
            file.write(reinterpret_cast<char*>(data.data()),size*sizeof(float));
        }
        for(auto state:layer->get_states())
        {
            size_t size=state->total_elements();
            std::vector<float> data(size);
            state->copy_to_host(data.data());
            file.write(reinterpret_cast<char*>(data.data()),size*sizeof(float));
        }
    }
    file.close();
}

void Network::load_weights(const std::string& filepath)
{
    std::ifstream file(filepath,std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file for loading weights!");
    for (auto layer:this->layers) 
    {
        for (auto weight:layer->get_weights()) 
        {
            size_t size = weight->total_elements();
            std::vector<float> data(size);
            file.read(reinterpret_cast<char*>(data.data()),size*sizeof(float));
            weight->copy_from_host(data.data());
        }
        for (auto state:layer->get_states()) 
        {
            size_t size = state->total_elements();
            std::vector<float> data(size);
            file.read(reinterpret_cast<char*>(data.data()),size*sizeof(float));
            state->copy_from_host(data.data());
        }
    }
    file.close();
}