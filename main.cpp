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
    net.add_layer(new Linear(1,64));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(64,64));
    net.add_layer(new ReLU());
    net.add_layer(new Linear(64,1));

    Optimizer* optimizer = new Adam(net.get_layers(),0.01f);
    Loss* loss=new MSELoss();

    net.compile(optimizer,loss);

    int size=128;
    std::vector<float> HX(size),HY(size);
    for(int i=0;i<size;i++)
    {
        float x=((float)rand()/RAND_MAX)*2.0f*3.14159f-3.14159f;
        HX[i]=x;HY[i]=sinf(x);
    }
    Tensor X=Tensor::zeros({size,1}),Y=Tensor::zeros({size,1});
    X.copy_from_host(HX.data());Y.copy_from_host(HY.data());
    
    for(int i=0;i<2000;i++)
    {
        float loss_=net.train_step(X,Y);
        if(i%100==0) std::cout<<"Epoch : " << i << " Loss : " << loss_ << std::endl;
    }

    std::cout << "\n--- Testing Predictions ---" << std::endl;
    
    // 1. Test Specific Known Values
    float test_vals[] = {-3.14159f / 2.0f, 0.0f, 3.14159f / 2.0f}; // -PI/2, 0, PI/2
    Tensor X_test = Tensor::zeros({3, 1});
    X_test.copy_from_host(test_vals);
    
    Tensor Y_pred = net.forward(X_test);
    float h_Y_pred[3];
    Y_pred.copy_to_host(h_Y_pred); // Bring predictions back to CPU
    
    std::cout << "sin(-PI/2) | True: -1.0 | Pred: " << h_Y_pred[0] << std::endl;
    std::cout << "sin(0)     | True:  0.0 | Pred: " << h_Y_pred[1] << std::endl;
    std::cout << "sin(PI/2)  | True:  1.0 | Pred: " << h_Y_pred[2] << std::endl;
    // 2. Calculate R^2 Score on the entire dataset
    Tensor Y_train_pred = net.forward(X);
    std::vector<float> h_Y_train_pred(size);
    Y_train_pred.copy_to_host(h_Y_train_pred.data());
    float ss_res = 0.0f;
    float ss_tot = 0.0f;
    float mean_y = 0.0f;
    
    // Calculate mean of true Y
    for(int i = 0; i < size; i++) mean_y += HY[i];
    mean_y /= size;
    // Calculate SS_res and SS_tot
    for(int i = 0; i < size; i++) {
        ss_res += (HY[i] - h_Y_train_pred[i]) * (HY[i] - h_Y_train_pred[i]);
        ss_tot += (HY[i] - mean_y) * (HY[i] - mean_y);
    }
    
    float r_squared = 1.0f - (ss_res / ss_tot);
    std::cout << "\nR^2 Score: " << r_squared << " (1.0 is a perfect fit)" << std::endl;
    std::cout << "\nSaving model weights..." << std::endl;
    net.save_weights("./weights/sine_model.bin");
    std::cout << "Model successfully saved to sine_model.bin!" << std::endl;
    return 0;
}
