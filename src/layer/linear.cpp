#include "../../include/layer/linear.h"
#include "../../include/core/tensor_ops.h"
#include <cmath>

Linear::Linear(int in,int out,Init init):W({in,out}),b({1,out}),dW({in,out}),db({1,out}),cached_X({0})
{
    if(init==Init::ZEROS) W=Tensor::zeros({in,out});    
    else if(init==Init::XAVIER) W=Tensor::randn({in,out}, 0.0f, (float)(sqrt(2.0f/(in+out))));
    else if (init==Init::KAIMING) W=Tensor::randn({in, out},0.0f,(float)(sqrt(2.0f/(in))));
    b=Tensor::zeros({1,out});
    dW=Tensor::zeros({in,out});
    db=Tensor::zeros({1,out});  
}

Tensor Linear::forward(const Tensor& X)
{
    this->cached_X=X;
    Tensor Y=X*W;
    add_bias(Y,b);
    return Y;
}

Tensor Linear::backward(const Tensor& dY)
{
    this->dW=multiply(this->cached_X,true,dY,false);
    this->db=sum_rows(dY);
    Tensor dX=multiply(dY,false,this->W,true);
    return dX;
}

std::vector<Tensor*> Linear::get_weights() {return {&this->W,&this->b};}
std::vector<Tensor*> Linear::get_grads() {return {&this->dW,&this->db};}