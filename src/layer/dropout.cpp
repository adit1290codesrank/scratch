#include "../../include/layer/dropout.h"
#include "../../include/core/dropout_ops.h"
#include <cstdlib>

Dropout::Dropout(float p):p(p),is_training(true){}

Tensor Dropout::forward(const Tensor& X)
{
    if(!is_training||p==0.0f) return X;

    mask=Tensor::zeros(X.shape);
    Tensor Y=Tensor::zeros(X.shape);

    unsigned int seed=rand();
    dropout_forward_gpu(X.get_data(),Y.get_data(),mask.get_data(),p,X.total_elements(),seed);

    return Y;
}

Tensor Dropout::backward(const Tensor& dY)
{
    if(!is_training||p==0.0f) return dY;

    Tensor dX=Tensor::zeros(dY.shape);
    dropout_backward_gpu(dY.get_data(),dX.get_data(),mask.get_data(),p,dY.total_elements());

    return dX;
}
