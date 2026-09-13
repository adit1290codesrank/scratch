#include "../../include/layer/linear.h"
#include "../../include/core/tensor_ops.h"
#include <cmath>

Linear::Linear(int in,int out,Init init,float custom_std,DType dt):W({in,out}),b({1,out}),dW({in,out}),db({1,out}),cached_X({0})
{
    Tensor Wf32({in,out});
    if(custom_std>0.0f) Wf32=Tensor::randn({in,out},0.0f,custom_std);
    else if(init==Init::ZEROS) Wf32=Tensor::zeros({in,out});
    else if(init==Init::XAVIER) Wf32=Tensor::randn({in,out},0.0f,(float)(sqrt(2.0f/(in+out))));
    else if(init==Init::KAIMING) Wf32=Tensor::randn({in,out},0.0f,(float)(sqrt(2.0f/in)));

    if(dt==DType::F32) W=Wf32;
    else { W=Tensor({in,out},dt); cast_tensor(W,Wf32); }

    Tensor bf32=Tensor::zeros({1,out});
    if(dt==DType::F32) b=bf32;
    else { b=Tensor({1,out},dt); cast_tensor(b,bf32); }

    dW=Tensor::zeros({in,out});
    db=Tensor::zeros({1,out});
}

// Linear::Linear(int in,int out,Init init,float custom_std):W({in,out}),b({1,out}),dW({in,out}),db({1,out}),cached_X({0})
// {
//     if(custom_std>0.0f) W=Tensor::randn({in,out},0.0f,custom_std);
//     else if(init==Init::ZEROS) W=Tensor::zeros({in,out});    
//     else if(init==Init::XAVIER) W=Tensor::randn({in,out},0.0f,(float)(sqrt(2.0f/(in+out))));
//     else if(init==Init::KAIMING) W=Tensor::randn({in,out},0.0f,(float)(sqrt(2.0f/in)));
//     b=Tensor::zeros({1,out});
//     dW=Tensor::zeros({in,out});
//     db=Tensor::zeros({1,out});  
// }


Tensor Linear::forward(const Tensor& X,const Tensor* pad_mask)
{
    this->cached_X=X;
    int d_in=X.shape.back();
    int batch_seq=X.total_elements()/d_in;

    Tensor X2d=X.reshape({batch_seq,d_in});
    Tensor Y2d=X2d*W;
    add_bias(Y2d,b);

    std::vector<int> out_shape=X.shape;
    out_shape.back()=W.shape.back();

    return Y2d.reshape(out_shape);
}

// Tensor Linear::forward(const Tensor& X,const Tensor* pad_mask)
// {
//     this->cached_X=X;
//     int d_in=X.shape.back();
//     int batch_seq=X.total_elements()/d_in;
    
//     Tensor X2d=X.reshape({batch_seq,d_in});
//     Tensor Y2d=X2d*W;
//     add_bias(Y2d,b);
    
//     std::vector<int> out_shape=X.shape;
//     out_shape.back()=W.shape.back();
    
//     return Y2d.reshape(out_shape);
// }

Tensor Linear::backward(const Tensor& dY)
{
    int d_out=dY.shape.back();
    int batch_seq=dY.total_elements()/d_out;
    int d_in=W.shape[0];

    Tensor dY2d=dY.reshape({batch_seq,d_out});
    Tensor X2d=this->cached_X.reshape({batch_seq,d_in});

    Tensor dW_raw=multiply(X2d,true,dY2d,false);
    Tensor db_raw=sum_rows(dY2d);
    if(dW_raw.dtype()==DType::F32) this->dW=dW_raw;
    else { this->dW=Tensor(dW_raw.shape,DType::F32); cast_tensor(this->dW,dW_raw); }
    if(db_raw.dtype()==DType::F32) this->db=db_raw;
    else { this->db=Tensor(db_raw.shape,DType::F32); cast_tensor(this->db,db_raw); }

    Tensor dX2d=multiply(dY2d,false,this->W,true);
    return dX2d.reshape(this->cached_X.shape);
}

// Tensor Linear::backward(const Tensor& dY)
// {
//     int d_out=dY.shape.back();
//     int batch_seq=dY.total_elements()/d_out;
//     int d_in=W.shape[0];
    
//     Tensor dY2d=dY.reshape({batch_seq,d_out});
//     Tensor X2d=this->cached_X.reshape({batch_seq,d_in});
    
//     this->dW=multiply(X2d,true,dY2d,false);
//     this->db=sum_rows(dY2d);
    
//     Tensor dX2d=multiply(dY2d,false,this->W,true);
//     return dX2d.reshape(this->cached_X.shape);
// }

std::vector<Tensor*> Linear::get_weights() {return {&this->W,&this->b};}
std::vector<Tensor*> Linear::get_grads() {return {&this->dW,&this->db};}