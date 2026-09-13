#pragma once
#include "tensor.h"

void layernorm_forward(const Tensor& X,const Tensor& g,const Tensor& b,Tensor& Y,int n,int d,float eps=1e-5f);
void layernorm_backward(const Tensor& dY,const Tensor& X,const Tensor& g,Tensor& dg,Tensor& db,Tensor& dX,int n,int d,float eps=1e-5f);
