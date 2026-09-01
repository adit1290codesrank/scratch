#include <iostream>
#include <iomanip>
#include <vector>
#include "include/core/network.h"
#include "include/layer/conv2d.h"
#include "include/layer/maxpool.h"
#include "include/layer/flatten.h"
#include "include/layer/linear.h"
#include "include/layer/activation.h"
#include "include/core/dataset.h"

int main()
{
    // 1. Build the exact same architecture
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

    // 2. Load the trained weights
    std::cout << "Loading trained weights from mnist_model.bin..." << std::endl;
    net.load_weights("weights/mnist_model_1.bin");

    // 3. Load the 10,000 image TEST Dataset
    Tensor X_test, Y_test;
    Dataset::load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", X_test, Y_test);

    int total_samples = X_test.shape[0];
    int batch_size = 128;
    int correct_predictions = 0;

    std::cout << "Evaluating on " << total_samples << " unseen test images..." << std::endl;

    for(int start = 0; start < total_samples; start += batch_size)
    {
        int end = start + batch_size > total_samples ? total_samples : start + batch_size;
        int current_batch = end - start;
        
        Tensor X_batch = X_test.slice(start, end);
        Tensor Y_batch = Y_test.slice(start, end);

        // Forward pass ONLY (No backward, no gradients)
        Tensor pred = net.forward(X_batch);

        // Download predictions and true labels to CPU RAM to calculate accuracy
        std::vector<float> h_pred(current_batch * 10);
        std::vector<float> h_Y(current_batch * 10);
        
        pred.copy_to_host(h_pred.data());
        Y_batch.copy_to_host(h_Y.data());

        for(int i = 0; i < current_batch; i++) {
            int best_pred = 0;
            float max_prob = -1.0f;
            
            int true_label = 0;
            float max_true = -1.0f;

            // Find Argmax for Prediction and Ground Truth
            for(int j = 0; j < 10; j++) {
                if(h_pred[i * 10 + j] > max_prob) {
                    max_prob = h_pred[i * 10 + j];
                    best_pred = j;
                }
                if(h_Y[i * 10 + j] > max_true) {
                    max_true = h_Y[i * 10 + j];
                    true_label = j;
                }
            }

            if(best_pred == true_label) {
                correct_predictions++;
            }
        }
    }

    float accuracy = ((float)correct_predictions / total_samples) * 100.0f;
    std::cout << "\n======================================" << std::endl;
    std::cout << "Test Accuracy: " << correct_predictions << " / " << total_samples << std::endl;
    std::cout << "Final Score:   " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}