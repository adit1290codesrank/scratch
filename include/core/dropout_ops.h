#pragma once
#include "tensor.h"

void dropout_forward_gpu(const Tensor& X,Tensor& Y,Tensor& mask,float p,unsigned int seed);
void dropout_backward_gpu(const Tensor& dY,Tensor& dX,const Tensor& mask,float p);
