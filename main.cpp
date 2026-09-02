#include <iostream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cstdlib>
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

int main()
{
    srand(time(NULL));

    Network net;
    net.add_layer(new Augment(false,2));

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


    Adam* optimizer=new Adam(net.get_layers(),0.005f);
    net.compile(optimizer,new CrossEntropyLoss());

    Tensor X_train,Y_train;
    Dataset::load_emnist("data/emnist-byclass-train-images-idx3-ubyte","data/emnist-byclass-train-labels-idx1-ubyte",X_train,Y_train,62);

    int batch_size=256,epochs=10,total=X_train.shape[0];
    std::cout << "Starting EMNIST ByClass ResNet Training on " << total << " images..." << std::endl;

    for(int i=0;i<=epochs;i++)
    {
        if (i==5) optimizer->set_lr(0.001f);
        if (i==8) optimizer->set_lr(0.0001f);

        float loss=0.0f,acc=0.0f;
        int b=0;
        for(int j=0;j<total;j+=batch_size)
        {
            int end=std::min(j+batch_size,total);
            if(end-j!=batch_size) continue;

            Tensor X_batch=X_train.slice(j,end),Y_batch=Y_train.slice(j,end);

            auto train_return=net.train_step(X_batch,Y_batch);
            loss+=train_return.first;acc+=train_return.second;
            b++;
            
            if(b%500==0) std::cout << "  Batch " << b << "/" << (total/batch_size) << " | Loss: " << std::fixed << std::setprecision(4) << (loss/b) << " | Acc: " << (acc/b)*100.0f << "%\r" << std::flush;
        }
        std::cout << "\nEpoch " << i << " Complete"<< " | Avg Loss: " << std::fixed << std::setprecision(4) << (loss/b)<< " | Avg Acc: " << (acc/b)*100.0f << "%" << std::endl;
        net.save_weights("weights/emnist_byclass_"+std::to_string(i)+".bin");
    }
}