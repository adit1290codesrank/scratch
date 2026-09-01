#include "../../include/layer/maxpool.h"
#include "../../include/core/conv_ops.h"

MaxPool::MaxPool(int k,int s):k(k),s(s){}

Tensor MaxPool::forward(const Tensor& X)
{
    this->cached_X=X;
    int n=X.shape[0],c=X.shape[1],h=X.shape[2],w=X.shape[3];
    int hout=(h-k)/s+1,wout=(w-k)/s+1;

    Tensor Y=Tensor::zeros({n,c,hout,wout});
    this->cached_mask=Tensor::zeros({n,c,hout,wout});

    maxpool_forward_gpu(X.get_data(),Y.get_data(),this->cached_mask.get_data(),n,c,h,w,hout,wout,k,s);
    return Y;
}

Tensor MaxPool::backward(const Tensor& dY)
{
    int n=cached_X.shape[0],c=cached_X.shape[1],h=cached_X.shape[2],w=cached_X.shape[3];
    int hout=(h-k)/s+1,wout=(w-k)/s+1;

    Tensor dX=Tensor::zeros(this->cached_X.shape);
    maxpool_backward_gpu(dY.get_data(),cached_mask.get_data(),dX.get_data(),n,c,h,w,hout,wout);
    return dX;
}