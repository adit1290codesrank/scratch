#include "../../include/layer/layernorm.h"
#include "../../include/core/layernorm_ops.h"

LayerNorm::LayerNorm(int dimension):dimension(dimension)
{
    g=Tensor::ones({dimension});
    b=Tensor::zeros({dimension});
    dg=Tensor::zeros({dimension});
    db=Tensor::zeros({dimension});
}

std::vector<Tensor*> LayerNorm::get_weights() { return {&g,&b}; }
std::vector<Tensor*> LayerNorm::get_grads() { return {&dg,&db}; }

Tensor LayerNorm::forward(const Tensor& X,const Tensor* pad_mask)
{
    this->cached_X=X;
    Tensor Y=Tensor::zeros(X.shape);
    layernorm_forward(X.get_data(),g.get_data(),b.get_data(),Y.get_data(),X.rows(),dimension);
    return Y;
}

Tensor LayerNorm::backward(const Tensor& dY)
{
    Tensor dX=Tensor::zeros(cached_X.shape);
    layernorm_backward(dY.get_data(),cached_X.get_data(),g.get_data(),dg.get_data(),db.get_data(),dX.get_data(),dY.rows(),dimension);
    return dX;
}
