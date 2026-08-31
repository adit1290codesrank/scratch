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

    int hin=X.shape[1],win=X.shape[2];
    int hout=(hin-k+2*p)/s+1,wout=(win-k+2*p)/s+1;

    this->cached_Xcol=Tensor::zeros({cin*k*k,hout*wout});
    im2col_gpu(X.get_data(),cin,hin,win,k,p,s,this->cached_Xcol.get_data());

    Tensor Yflat=W*cached_Xcol;
    add_bias_conv(Yflat,b);
    return Yflat.reshape({cout,hout,wout});
}

Tensor Conv2D::backward(const Tensor& dY)
{
    int hin=cached_X.shape[1],win=cached_X.shape[2];

    Tensor dYflat=dY.reshape({cout,dY.shape[1]*dY.shape[2]});
    this->db=sum_spatial(dYflat);
    this->dW=multiply(dYflat,false,this->cached_Xcol,true);

    Tensor dX_col=multiply(this->W,true,dYflat,false);
    Tensor dX=Tensor::zeros(this->cached_X.shape);
    col2im_gpu(dX_col.get_data(),cin,hin,win,k,p,s,dX.get_data());
    return dX;
}
