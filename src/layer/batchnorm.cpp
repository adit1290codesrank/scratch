#include "../../include/layer/batchnorm.h"
#include "../../include/core/batchnorm_ops.h"

BatchNorm::BatchNorm(int n,float e,float m):n(n),e(e),m(m),is_training(true)
{
    g=Tensor::ones({n});b=Tensor::zeros({n});dg=Tensor::zeros({n});db=Tensor::zeros({n});rm=Tensor::zeros({n});rv=Tensor::ones({n});
}

Tensor BatchNorm::forward(const Tensor& X)
{
    cached_X=X;
    int n=X.shape[0];
    int c=(X.shape.size()>1)?X.shape[1]:1;
    int s=X.total_elements()/(n*c);

    Tensor Y=Tensor::zeros(X.shape);
    cached_X_=Tensor::zeros(X.shape);

    if(is_training)
    {
        cached_bmean=Tensor::zeros({c});cached_bvar=Tensor::zeros({c});
        bn_param_gpu(X.get_data(),cached_bmean.get_data(),cached_bvar.get_data(),n,c,s);
        bn_running_gpu(rm.get_data(),rv.get_data(),cached_bmean.get_data(),cached_bvar.get_data(),m,c);
        bn_forward_gpu(X.get_data(),cached_bmean.get_data(),cached_bvar.get_data(),g.get_data(),b.get_data(),Y.get_data(),cached_X_.get_data(),n,c,s,e);
    }
    else bn_forward_gpu(X.get_data(),rm.get_data(),rv.get_data(),g.get_data(),b.get_data(),Y.get_data(),nullptr,n,c,s,e);
    return Y;
}

Tensor BatchNorm::backward(const Tensor& dY)
{
    int n=cached_X.shape[0];
    int c=(cached_X.shape.size()>1)?cached_X.shape[1]:1;
    int s=cached_X.total_elements()/(n*c);

    Tensor dX=Tensor::zeros(cached_X.shape);

    bn_backwards_param_gpu(dY.get_data(),cached_X_.get_data(),dg.get_data(),db.get_data(),n,c,s);
    bn_backward_gpu(dY.get_data(),cached_X_.get_data(),cached_bvar.get_data(),g.get_data(),dX.get_data(),dg.get_data(),db.get_data(),n,c,s,e);
    return dX;
}