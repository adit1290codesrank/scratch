#include "../../include/layer/gap.h"

Tensor GAP::forward(const Tensor& X)
{
    n=X.shape[0];c=X.shape[1];h=X.shape[2];w=X.shape[3];

    Tensor Y=Tensor::zeros({n,c});
    gap_forward_gpu(X.get_data(),Y.get_data(),n,c,h*w);
    return Y;
}

Tensor GAP::backward(const Tensor& dY)
{
    Tensor dX=Tensor::zeros({n,c,h,w});
    gap_backward_gpu(dY.get_data(),dX.get_data(),n,c,h*w);
    return dX;
}