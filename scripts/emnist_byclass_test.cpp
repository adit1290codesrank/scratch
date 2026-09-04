#include <iostream>
#include <iomanip>
#include <vector>
#include "include/core/network.h"
#include "include/layer/conv2d.h"
#include "include/layer/maxpool.h"
#include "include/layer/gap.h"
#include "include/layer/linear.h"
#include "include/layer/activation.h"
#include "include/layer/batchnorm.h"
#include "include/layer/res.h"
#include "include/core/dataset.h"
#include "include/core/memory.h"

int main()
{
    Network net;

    net.add_layer(new Conv2D(1,32,3,1,1));
    net.add_layer(new BatchNorm(32));
    net.add_layer(new ReLU());

    Res* res1=new Res();
    res1->add_main(new Conv2D(32,32,3,1,1));
    res1->add_main(new BatchNorm(32));
    res1->add_main(new ReLU());
    res1->add_main(new Conv2D(32,32,3,1,1));
    res1->add_main(new BatchNorm(32));
    net.add_layer(res1);
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2,2));

    net.add_layer(new Conv2D(32,64,3,1,1));
    net.add_layer(new BatchNorm(64));
    net.add_layer(new ReLU());

    Res* res2=new Res();
    res2->add_main(new Conv2D(64,64,3,1,1));
    res2->add_main(new BatchNorm(64));
    res2->add_main(new ReLU());
    res2->add_main(new Conv2D(64,64,3,1,1));
    res2->add_main(new BatchNorm(64));
    net.add_layer(res2);
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2,2));

    net.add_layer(new Conv2D(64,128,3,1,1));
    net.add_layer(new BatchNorm(128));
    net.add_layer(new ReLU());

    net.add_layer(new GAP());
    net.add_layer(new Linear(128,62));
    net.add_layer(new Softmax());

    std::cout << "Loading trained weights..." << std::endl;

    net.load_weights("weights/emnist_byclass_9.bin");

    Tensor X_test,Y_test;
    Dataset::load_emnist("data/emnist-byclass-test-images-idx3-ubyte","data/emnist-byclass-test-labels-idx1-ubyte",X_test,Y_test,62);

    int total_images=X_test.shape[0],batch_size=256,correct=0,evaluated=0;

    for (int start = 0; start < total_images; start += batch_size)
    {
        int end = std::min(start + batch_size, total_images);
        if (end - start != batch_size) continue;
        Tensor X_batch = X_test.slice(start, end);
        Tensor Y_batch = Y_test.slice(start, end);
        Tensor pred = net.forward(X_batch);
        
        int classes = 62;
        std::vector<float> hpred(batch_size * classes), hy(batch_size * classes);
        pred.copy_to_host(hpred.data());
        Y_batch.copy_to_host(hy.data());
        for (int i = 0; i < batch_size; i++) {
            int best_p = 0, best_y = 0;
            float max_p = -1e9, max_y = -1e9;
            for (int j = 0; j < classes; j++) {
                if (hpred[i * classes + j] > max_p) { max_p = hpred[i * classes + j]; best_p = j; }
                if (hy[i * classes + j] > max_y) { max_y = hy[i * classes + j]; best_y = j; }
            }
            if (best_p == best_y) correct++;
        }
        evaluated += batch_size;
        
        if (start % (batch_size * 20) == 0) {
            std::cout << "Evaluated " << evaluated << " / " << total_images << "\r" << std::flush;
        }
    }
    std::cout << "\n\n=========================================\n";
    std::cout << "Final Test Accuracy: " << ((float)correct / evaluated) * 100.0f << "%\n";
    std::cout << "=========================================\n";
    clear_memory_pool();
    return 0;
}