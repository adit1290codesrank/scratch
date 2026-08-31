#include <iostream>
#include "include/core/network.h"
#include "include/layer/linear.h"
#include "include/layer/activation.h"
#include "include/core/loss.h"
#include "include/core/optimizer.h"

int main() 
{
    std::cout << "Neural Network Test"<<std::endl;

    Network net;
    net.add_layer(new Linear(10,32));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(32,2));
    
    Optimizer* adam=new Adam(net.get_layers(),0.01f);
    Loss* mse=new MSELoss();
    net.compile(adam,mse);

    Tensor X=Tensor::zeros({1,10}); 
    Tensor Y=Tensor::zeros({1,2}); 

    std::cout<<"Running forward and backward pass"<<std::endl;
    float loss=net.train_step(X, Y);
    
    std::cout<<"Initial Loss: "<<loss<<std::endl;

    return 0;
}
