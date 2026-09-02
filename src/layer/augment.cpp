#include "../../include/layer/augment.h"
#include <vector>
#include <cstdlib>

Augment::Augment(bool flip,int max):is_training(true),flip(flip),max(max),pgpu(nullptr),b_(0){}

Augment::~Augment(){free_augment(pgpu);}

Tensor Augment::forward(const Tensor& X)
{
    if(!is_training) return X;

    int b=X.shape[0],c=X.shape[1],h=X.shape[2],w=X.shape[3];
    
    if(b!=b_) 
    {
        free_augment(pgpu);
        pgpu=alloc_augment(b);
        b_=b;
    }

    std::vector<AugmentParams> p(b);
    for(int i=0;i<b;i++)
    {
        p[i].flip=flip?(rand()%2):0;
        p[i].dx=(rand()%(2*max+1))-max;
        p[i].dy=(rand()%(2*max+1))-max;
    }

    Tensor Y=Tensor::zeros(X.shape);
    augment_gpu(X.get_data(),Y.get_data(),pgpu,p.data(),b,c,h,w);
    return Y;
}

Tensor Augment::backward(const Tensor& dY){return dY;}