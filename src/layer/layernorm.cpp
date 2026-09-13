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
    Tensor Y=Tensor::zeros(X.shape,X.dtype());
    layernorm_forward(X,g,b,Y,X.total_elements()/dimension,dimension);
    return Y;
}

Tensor LayerNorm::backward(const Tensor& dY)
{
    Tensor dX=Tensor::zeros(cached_X.shape,cached_X.dtype());
    layernorm_backward(dY,cached_X,g,dg,db,dX,dY.total_elements()/dimension,dimension);
    return dX;
}
