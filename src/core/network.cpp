#include "../../include/core/network.h"
#include <string>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

void Network::add_layer(Layer* layer){this->layers.push_back(layer);}

std::vector<Layer*> Network::get_layers(){return layers;}

void Network::compile(Optimizer* optimizer,Loss* loss){this->optimizer=optimizer;this->loss=loss;}

Tensor Network::forward(Tensor X){for(auto layer:layers)X=layer->forward(X);return X;}

void Network::backward(Tensor dY){for(int i=layers.size()-1;i>=0;i--)dY=layers[i]->backward(dY);}

float accuracy(std::vector<float>& hpred,std::vector<float>& hy,int batch_size,int classes)
{
    int correct=0;
    for (int i=0;i<batch_size;i++) 
    {
        int best_p=0,best_y=0;
        float max_p=-1e9,max_y=-1e9;
        
        for (int j=0;j<classes;j++) 
        {
            if (hpred[i*classes+j]>max_p){max_p=hpred[i*classes+j];best_p=j;}
            if (hy[i*classes+j]>max_y){max_y=hy[i*classes+j];best_y=j;}
        }
        if(best_p==best_y) correct++;
    }
    float accuracy=(float)correct/batch_size;
    return accuracy;
}

std::pair<float,float> Network::train_step(const Tensor& X,const Tensor& Y)
{
    Tensor pred=this->forward(X);
    float loss=this->loss->calculate_loss(pred,Y);
    int batch_size=Y.shape[0],classes=Y.shape[1];
    std::vector<float> hpred(batch_size*classes),hy(batch_size*classes);
    pred.copy_to_host(hpred.data());Y.copy_to_host(hy.data());
    float acc=accuracy(hpred,hy,batch_size,classes);
    Tensor dY=this->loss->backward_loss(pred, Y);
    this->backward(dY);
    this->optimizer->step();
    return {loss,acc};
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