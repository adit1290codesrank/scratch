#pragma once
#include "tensor.h"

Tensor multiply(const Tensor& a,bool transA,const Tensor& b,bool transB);
Tensor add(const Tensor& a,const Tensor& b);
void add_bias(Tensor& Y,const Tensor& b);
Tensor sum_rows(const Tensor& dY);
void add_bias_conv(Tensor& Yflat,const Tensor& b);
Tensor sum_spatial(const Tensor& dYflat);

Tensor multiply_conv_forward(const Tensor& W,const Tensor& Xcol);
Tensor multiply_conv_backward_dX(const Tensor& W,const Tensor& dYflat);
void multiply_conv_backward_dW(Tensor& dW,const Tensor& dYflat,const Tensor& Xcol);