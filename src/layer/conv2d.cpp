#include "../../include/layer/conv2d.h"
#include "../../include/core/tensor_ops.h"
#include "../../include/core/conv_ops.h"
#include <cmath>

Conv2D::Conv2D(int cin,int cout,int k,int s,int p,Init init):cin(cin),cout(cout),k(k),s(s),p(p)
{
    int r=cout*k*k,c=cin*k*k;
    if(init==Init::ZEROS) W=Tensor::zeros({cout,c});
    else if(init==Init::KAIMING) W=Tensor::randn({cout,c},0.0f,(float)(sqrt(2.0f/c)));
    else if(init==Init::XAVIER) W=Tensor::randn({cout,c},0.0f,(float)(sqrt(2.0f/(r+c))));

    b=Tensor::zeros({1,cout});
    dW=Tensor::zeros({cout,c});
    db=Tensor::zeros({1,cout});
}

std::vector<Tensor*> Conv2D::get_weights(){return {&this->W,&this->b};}
std::vector<Tensor*> Conv2D::get_grads(){return {&this->dW,&this->db};}


Tensor Conv2D::forward(const Tensor& X)
{
    this->cached_X=X;

    int batch=X.shape[0],hin=X.shape[2],win=X.shape[3];
    int hout=(hin-k+2*p)/s+1,wout=(win-k+2*p)/s+1;

    this->cached_Xcol=Tensor::zeros({batch,cin*k*k,hout*wout});
    im2col_gpu(X.get_data(),batch,cin,hin,win,k,p,s,this->cached_Xcol.get_data());

    Tensor Yflat=multiply_conv_forward(this->W,this->cached_Xcol);
    add_bias_conv(Yflat,b);
    return Yflat.reshape({batch,cout,hout,wout});
}

Tensor Conv2D::backward(const Tensor& dY)
{
    int batch=cached_X.shape[0],hin=cached_X.shape[2],win=cached_X.shape[3];

    Tensor dYflat=dY.reshape({batch,cout,dY.shape[2]*dY.shape[3]});
    this->db=sum_spatial(dYflat);
    multiply_conv_backward_dW(this->dW,dYflat,this->cached_Xcol);

    Tensor dX_col=multiply_conv_backward_dX(this->W,dYflat);
    Tensor dX=Tensor::zeros(this->cached_X.shape);
    col2im_gpu(dX_col.get_data(),batch,cin,hin,win,k,p,s,dX.get_data());
    return dX;
}
