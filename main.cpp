#include <iostream>
#include <vector>
#include <string>
#include "include/core/network.h"
#include "include/layer/conv2d.h"
#include "include/layer/maxpool.h"
#include "include/layer/gap.h"
#include "include/layer/linear.h"
#include "include/layer/activation.h"
#include "include/layer/batchnorm.h"
#include "include/layer/res.h"
#include "include/layer/augment.h"
#include "include/core/dataset.h"
#include "include/core/memory.h"
#include "include/layer/dropout.h"

// Helper function to calculate accuracy
float calc_accuracy(const std::vector<float>& hpred, const std::vector<float>& hy, int batch_size, int classes) {
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {
        int best_p = 0, best_y = 0;
        float max_p = -1e9, max_y = -1e9;
        for (int j = 0; j < classes; j++) {
            if (hpred[i * classes + j] > max_p) { max_p = hpred[i * classes + j]; best_p = j; }
            if (hy[i * classes + j] > max_y) { max_y = hy[i * classes + j]; best_y = j; }
        }
        if (best_p == best_y) correct++;
    }
    return (float)correct / batch_size;
}

int main()
{
    Network net;

    net.add_layer(new Augment(true,4,8));

    // 2. Exact same architecture to match the weights
    net.add_layer(new Conv2D(3, 32, 3, 1, 1));
    net.add_layer(new BatchNorm(32));
    net.add_layer(new ReLU());

    Res* res1 = new Res();
    res1->add_main(new Conv2D(32, 32, 3, 1, 1));
    res1->add_main(new BatchNorm(32));
    res1->add_main(new ReLU());
    res1->add_main(new Conv2D(32, 32, 3, 1, 1));
    res1->add_main(new BatchNorm(32));
    net.add_layer(res1);
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2, 2));

    net.add_layer(new Conv2D(32, 64, 3, 1, 1));
    net.add_layer(new BatchNorm(64));
    net.add_layer(new ReLU());

    Res* res2 = new Res();
    res2->add_main(new Conv2D(64, 64, 3, 1, 1));
    res2->add_main(new BatchNorm(64));
    res2->add_main(new ReLU());
    res2->add_main(new Conv2D(64, 64, 3, 1, 1));
    res2->add_main(new BatchNorm(64));
    net.add_layer(res2);
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2, 2));

    net.add_layer(new Conv2D(64, 128, 3, 1, 1));
    net.add_layer(new BatchNorm(128));
    net.add_layer(new ReLU());

    Res* res3 = new Res();
    res3->add_main(new Conv2D(128, 128, 3, 1, 1));
    res3->add_main(new BatchNorm(128));
    res3->add_main(new ReLU());
    res3->add_main(new Conv2D(128, 128, 3, 1, 1));
    res3->add_main(new BatchNorm(128));
    net.add_layer(res3);
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2, 2));

    net.add_layer(new Conv2D(128, 256, 3, 1, 1));
    net.add_layer(new BatchNorm(256));
    net.add_layer(new ReLU());

    Res* res4 = new Res();
    res4->add_main(new Conv2D(256, 256, 3, 1, 1));
    res4->add_main(new BatchNorm(256));
    res4->add_main(new ReLU());
    res4->add_main(new Conv2D(256, 256, 3, 1, 1));
    res4->add_main(new BatchNorm(256));
    net.add_layer(res4);
    net.add_layer(new ReLU());

    net.add_layer(new GAP());
    net.add_layer(new Linear(256,128));
    net.add_layer(new BatchNorm(128));
    net.add_layer(new ReLU());
    net.add_layer(new Dropout(0.3f));
    net.add_layer(new Linear(128,10));
    net.add_layer(new Softmax());

    // 3. LOAD THE WEIGHTS (Change this to whatever epoch you downloaded!)
    std::string weight_file = "weights/cifar10_byclass_33.bin";
    std::cout << "Loading weights from: " << weight_file << std::endl;
    net.load_weights(weight_file);

    // 4. LOAD THE TEST DATASET (Notice the 'true' flag at the end!)
    Tensor X_test, Y_test;
    Dataset::load_cifar10("data/cifar-10-batches-bin", X_test, Y_test, true);

    int total_images = X_test.shape[0];
    int batch_size = 256;
    float epoch_acc = 0.0f;
    int batches = 0;

    std::cout << "Starting Evaluation on " << total_images << " Test Images..." << std::endl;
    net.eval();

    for (int start = 0; start < total_images; start += batch_size)
    {
        int end = std::min(start + batch_size, total_images);
        if(end - start != batch_size) continue; 
        
        Tensor X_batch = X_test.slice(start, end);
        Tensor Y_batch = Y_test.slice(start, end);

        // Forward Pass ONLY! No backprop.
        Tensor pred = net.forward(X_batch);
        
        // Calculate Accuracy
        std::vector<float> hpred(batch_size * 10), hy(batch_size * 10);
        pred.copy_to_host(hpred.data());
        Y_batch.copy_to_host(hy.data());
        
        float acc = calc_accuracy(hpred, hy, batch_size, 10);
        epoch_acc += acc;
        batches++;

        std::cout << "  Testing Batch " << batches << " | Acc: " << acc * 100.0f << "%\r" << std::flush;
    }

    std::cout << "\n\nFINAL TEST ACCURACY: " << (epoch_acc / batches) * 100.0f << "%!" << std::endl;

    clear_memory_pool();
    return 0;
}