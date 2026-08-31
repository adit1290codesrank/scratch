#include "../../include/layer/activation.h"
#include "../../include/core/activation_ops.h" 

Tensor ReLU::forward(const Tensor& X)
{
    this->cached_X=X;
    Tensor Y=X.clone();
    relu_forward(Y);
    return Y;
}

Tensor ReLU::backward(const Tensor& dY)
{
    Tensor dX(dY.shape);
    relu_backward(dY,this->cached_X,dX);
    return dX;
}

Tensor Softmax::forward(const Tensor& X)
{
    this->cached_X=X;
    Tensor Y=X.clone();
    softmax_forward(Y);
    return Y;
}

Tensor Softmax::backward(const Tensor& dY){return dY;}