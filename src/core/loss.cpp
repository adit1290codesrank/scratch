#include "../../include/core/loss.h"
#include "../../include/core/loss_ops.h"
#include <stdexcept>

float MSELoss::calculate_loss(const Tensor& pred,const Tensor& target)
{
    if(pred.shape!=target.shape) throw std::invalid_argument("Shapes must match for MSELoss");
    return mse_forward(pred,target);
}

Tensor MSELoss::backward_loss(const Tensor& pred,const Tensor& target)
{
    Tensor dY(pred.shape);
    mse_backward(pred,target,dY);
    return dY;
}

float CrossEntropyLoss::calculate_loss(const Tensor& pred,const Tensor& target)
{
    if(pred.shape!=target.shape) throw std::invalid_argument("Shapes must match for CrossEntropyLoss");
    return ce_forward(pred, target);
}

Tensor CrossEntropyLoss::backward_loss(const Tensor& pred,const Tensor& target)
{
    Tensor dY(pred.shape);
    ce_backward(pred, target,dY);
    return dY;
}

LSCrossEntropyLoss::LSCrossEntropyLoss(int n,float a):n(n),a(a){}

float LSCrossEntropyLoss::calculate_loss(const Tensor& pred,const Tensor& target)
{
    if(pred.shape!=target.shape) throw std::invalid_argument("Shapes must match for CrossEntropyLoss");
    return ls_ce_forward(pred, target,this->n,this->a);
}

Tensor LSCrossEntropyLoss::backward_loss(const Tensor& pred,const Tensor& target)
{
    Tensor dY(pred.shape);
    ls_ce_backward(pred, target,dY,this->n,this->a);
    return dY;
}