#include "../../include/layer/dropout.h"
#include "../../include/core/dropout_ops.h"
#include <cstdlib>

Dropout::Dropout(float p):p(p),is_training(true){}

Tensor Dropout::forward(const Tensor& X,const Tensor* pad_mask)
{
    if(!is_training||p==0.0f) return X;

    mask=Tensor::zeros(X.shape,X.dtype());
    Tensor Y=Tensor::zeros(X.shape,X.dtype());

    unsigned int seed=rand();
    dropout_forward_gpu(X,Y,mask,p,seed);

    return Y;
}

Tensor Dropout::backward(const Tensor& dY)
{
    if(!is_training||p==0.0f) return dY;

    Tensor dX=Tensor::zeros(dY.shape,dY.dtype());
    dropout_backward_gpu(dY,dX,mask,p);
    return dX;
}
