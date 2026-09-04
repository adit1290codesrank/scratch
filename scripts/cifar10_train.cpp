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
#include "include/layer/dropout.h"
#include "include/core/dataset.h"
#include "include/core/memory.h"
#include "include/core/scheduler.h"

int main()
{
    srand(time(NULL));

    Network net;

    net.add_layer(new Augment(true,4));

    net.add_layer(new Conv2D(3,64,3,1,1));
    net.add_layer(new BatchNorm(64));
    net.add_layer(new ReLU());

    Res* res1=new Res();
    res1->add_main(new Conv2D(64,64,3,1,1));
    res1->add_main(new BatchNorm(64));
    res1->add_main(new ReLU());
    res1->add_main(new Conv2D(64,64,3,1,1));
    res1->add_main(new BatchNorm(64));
    net.add_layer(res1);
    net.add_layer(new ReLU());

    Res* pres2=new Res();
    pres2->add_main(new Conv2D(64,128,3,2,1));
    pres2->add_main(new BatchNorm(128));
    pres2->add_main(new ReLU());
    pres2->add_main(new Conv2D(128,128,3,1,1));
    pres2->add_main(new BatchNorm(128));
    pres2->add_skip(new Conv2D(64,128,1,2,0));
    pres2->add_skip(new BatchNorm(128));
    net.add_layer(pres2);
    net.add_layer(new ReLU());

    Res* res2=new Res();
    res2->add_main(new Conv2D(128,128,3,1,1));
    res2->add_main(new BatchNorm(128));
    res2->add_main(new ReLU());
    res2->add_main(new Conv2D(128,128,3,1,1));
    res2->add_main(new BatchNorm(128));
    net.add_layer(res2);
    net.add_layer(new ReLU());

    Res* pres3=new Res();
    pres3->add_main(new Conv2D(128,256,3,2,1));
    pres3->add_main(new BatchNorm(256));
    pres3->add_main(new ReLU());
    pres3->add_main(new Conv2D(256,256,3,1,1));
    pres3->add_main(new BatchNorm(256));
    pres3->add_skip(new Conv2D(128,256,1,2,0));
    pres3->add_skip(new BatchNorm(256));
    net.add_layer(pres3);
    net.add_layer(new ReLU());

    Res* res3=new Res();
    res3->add_main(new Conv2D(256,256,3,1,1));
    res3->add_main(new BatchNorm(256));
    res3->add_main(new ReLU());
    res3->add_main(new Conv2D(256,256,3,1,1));
    res3->add_main(new BatchNorm(256));
    net.add_layer(res3);
    net.add_layer(new ReLU());

    Res* pres4=new Res();
    pres4->add_main(new Conv2D(256,512,3,2,1));
    pres4->add_main(new BatchNorm(512));
    pres4->add_main(new ReLU());
    pres4->add_main(new Conv2D(512,512,3,1,1));
    pres4->add_main(new BatchNorm(512));
    pres4->add_skip(new Conv2D(256,512,1,2,0));
    pres4->add_skip(new BatchNorm(512));
    net.add_layer(pres4);
    net.add_layer(new ReLU());

    Res* res4=new Res();
    res4->add_main(new Conv2D(512,512,3,1,1));
    res4->add_main(new BatchNorm(512));
    res4->add_main(new ReLU());
    res4->add_main(new Conv2D(512,512,3,1,1));
    res4->add_main(new BatchNorm(512));
    net.add_layer(res4);
    net.add_layer(new ReLU());

    net.add_layer(new GAP());
    net.add_layer(new Dropout(0.3f));
    net.add_layer(new Linear(512,10));
    net.add_layer(new Softmax());

    int epochs=80;

    Adam* optimizer=new Adam(net.get_layers(),0.005f,0.9f,0.999f,1e-8f,1e-3f);
    CosineAnnealing* scheduler=new CosineAnnealing(optimizer,0.005f,0.0001f,epochs);
    net.compile(optimizer,new LSCrossEntropyLoss(10,0.1f));

    Tensor X_train,Y_train,X_test,Y_test;
    Dataset::load_cifar10("data/cifar-10-batches-bin",X_train,Y_train,false);
    Dataset::load_cifar10("data/cifar-10-batches-bin",X_test,Y_test,true);

    int total=X_train.shape[0],test_total=X_test.shape[0],batch_size=128;

    std::cout << "Starting CIFAR-10 ResNet Training on " << total << " images..." << std::endl;

    for(int i=1;i<=epochs;i++)
    {
        scheduler->step();
        float loss=0.0f,acc=0.0f,val_acc=0.0f;
        int b=0,val_b=0;

        net.train();
        for(int j=0;j<total;j+=batch_size)
        {
            int end=std::min(j+batch_size,total);
            if(end-j!=batch_size)continue;

            Tensor X_batch=X_train.slice(j,end),Y_batch=Y_train.slice(j,end);
            
            auto train_return=net.train_step(X_batch,Y_batch);
            loss+=train_return.first;acc+=train_return.second;
            b++;
            if(b%50==0) std::cout<<"  Batch "<<b<<"/"<<(total/batch_size)<<" | Loss: "<<std::fixed<<std::setprecision(4)<<(loss/b)<<" | Acc: "<<(acc/b)*100.0f<<"%\r"<<std::flush;
        }

        net.eval();
        for(int j=0;j<test_total;j+=batch_size)
        {
            int end=std::min(j+batch_size,test_total);
            if(end-j!=batch_size)continue;
            
            Tensor X_batch=X_test.slice(j,end),Y_batch=Y_test.slice(j,end);
            Tensor pred=net.forward(X_batch);
            
            std::vector<float> hpred(batch_size*10),hy(batch_size*10);
            pred.copy_to_host(hpred.data());
            Y_batch.copy_to_host(hy.data());
            
            int correct=0;
            for(int k=0;k<batch_size;k++)
            {
                int best_p=0,best_y=0;
                float max_p=-1e9,max_y=-1e9;
                for(int c=0;c<10;c++)
                {
                    if(hpred[k*10+c]>max_p){max_p=hpred[k*10+c];best_p=c;}
                    if(hy[k*10+c]>max_y){max_y=hy[k*10+c];best_y=c;}
                }
                if(best_p==best_y)correct++;
            }
            val_acc+=(float)correct/batch_size;
            val_b++;
        }

        std::cout<<"\nEpoch "<<i<<" Complete | Train Loss: "<<std::fixed<<std::setprecision(4)<<(loss/b)<<" | Train Acc: "<<(acc/b)*100.0f<<"% | Val Acc: "<<(val_acc/val_b)*100.0f<<"%"<<std::endl;
        net.save_weights("weights/cifar10_"+std::to_string(i)+".bin");
    }
    clear_memory_pool();
    return 0;
}