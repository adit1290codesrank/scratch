#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
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
#include "include/core/scheduler.h"

int main()
{
    srand(time(NULL));

    Network net;

    net.add_layer(new Augment(true,4));

    net.add_layer(new Conv2D(3,32,3,1,1));
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
    net.add_layer(new MaxPool(2, 2));

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

    Res* res3=new Res();
    res3->add_main(new Conv2D(128,128,3,1,1));
    res3->add_main(new BatchNorm(128));
    res3->add_main(new ReLU());
    res3->add_main(new Conv2D(128,128,3,1,1));
    res3->add_main(new BatchNorm(128));
    net.add_layer(res3);
    net.add_layer(new ReLU());
    net.add_layer(new MaxPool(2, 2));

    net.add_layer(new Conv2D(128,256,3,1,1));
    net.add_layer(new BatchNorm(256));
    net.add_layer(new ReLU());

    Res* res4=new Res();
    res4->add_main(new Conv2D(256,256,3,1,1));
    res4->add_main(new BatchNorm(256));
    res4->add_main(new ReLU());
    res4->add_main(new Conv2D(256,256,3,1,1));
    res4->add_main(new BatchNorm(256));
    net.add_layer(res4);
    net.add_layer(new ReLU());

    net.add_layer(new GAP());
    net.add_layer(new Linear(256,10));
    net.add_layer(new Softmax());

    int epochs=50;

    Adam* optimizer=new Adam(net.get_layers(),0.005f);
    CosineAnnealing* scheduler=new CosineAnnealing(optimizer,0.005f,0.0001f,epochs);
    net.compile(optimizer,new CrossEntropyLoss());

    Tensor X_train,Y_train;
    Dataset::load_cifar10("data/cifar-10-batches-bin",X_train,Y_train,false);

    int total=X_train.shape[0],batch_size=256;

    std::cout << "Starting CIFAR-10 ResNet Training on " << total << " images..." << std::endl;

    for(int i=1;i<=epochs;i++)
    {
        scheduler->step();
        float loss=0.0f,acc=0.0f;
        int b=0;

        for(int j=0;j<total;j+=batch_size)
        {
            int end=std::min(j+batch_size,total);
            if(end-j!=batch_size)continue;

            Tensor X_batch=X_train.slice(j,end),Y_batch=Y_train.slice(j,end);
            
            auto train_return=net.train_step(X_batch,Y_batch);
            loss+=train_return.first;acc+=train_return.second;
            b++;
            if(b%500==0) std::cout << "  Batch " << b << "/" << (total/batch_size) << " | Loss: " << std::fixed << std::setprecision(4) << (loss/b) << " | Acc: " << (acc/b)*100.0f << "%\r" << std::flush;
        }
        std::cout << "\nEpoch " << i << " Complete"<< " | Avg Loss: " << std::fixed << std::setprecision(4) << (loss/b)<< " | Avg Acc: " << (acc/b)*100.0f << "%" << std::endl;
        net.save_weights("weights/cifar10_byclass_"+std::to_string(i)+".bin");
    }
    clear_memory_pool();
    return 0;
}