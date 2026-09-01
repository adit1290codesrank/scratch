#include <iostream>
#include "include/core/network.h"
#include "include/layer/conv2d.h"
#include "include/layer/maxpool.h"
#include "include/layer/flatten.h"
#include "include/layer/linear.h"
#include "include/layer/activation.h"
#include "include/core/loss.h"
#include "include/core/optimizer.h"
#include "include/core/dataset.h"


int main()
{
    Tensor X_train,Y_train;
    Dataset::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",X_train,Y_train);

    Network net;
    net.add_layer(new Conv2D(1,8,3,1,1));
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2,2));

    net.add_layer(new Conv2D(8,16,3,1,1));
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2,2));

    net.add_layer(new Flatten());
    net.add_layer(new Linear(784,10));
    net.add_layer(new Softmax());

    Optimizer* optimizer=new Adam(net.get_layers(),0.005f);
    Loss* loss=new CrossEntropyLoss();
    net.compile(optimizer,loss);

    int total=X_train.shape[0],batch=128,epochs=5;

    std::cout<<"Starting MNIST Training..."<<std::endl;
    for(int i=0;i<epochs;i++)
    {
        float loss_=0.0f;
        int steps=0;

        for(int j=0;j<total;j+=batch)
        {
            int end=j+batch>total?total:j+batch;

            Tensor X_batch=X_train.slice(j,end),Y_batch=Y_train.slice(j,end);
            float l=net.train_step(X_batch,Y_batch);
            loss_+=l;
            steps++;
            if(steps%100==0) std::cout<<"Epoch: "<<i<<" | Step: "<<steps<<"/468 | Loss: "<<l<<std::endl;
        }
        std::cout<<"--- Epoch "<<i<<" Average Loss: "<<(loss_/steps)<<" ---"<<std::endl;
    }
    
    std::cout<<"Saving model weights for inference..."<<std::endl;
    net.save_weights("weights/mnist_model_1.bin");
    std::cout<<"Weights saved to mnist_model.bin!"<<std::endl;

    return 0;
}