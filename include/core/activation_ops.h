#pragma once
#include "tensor.h"

void relu_forward(Tensor& Y);
void relu_backward(const Tensor& dY,const Tensor& cached_X,Tensor& dX);
void softmax_forward(Tensor& X);