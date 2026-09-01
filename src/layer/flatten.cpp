#include "../../include/layer/flatten.h"

Tensor Flatten::forward(const Tensor& X)
{
    this->cached_shape=X.shape;
    return X.reshape({X.shape[0],X.shape[1]*X.shape[2]*X.shape[3]});
}

Tensor Flatten::backward(const Tensor& dY){return dY.reshape(this->cached_shape);}