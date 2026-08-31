#pragma once
#include "tensor.h"

Tensor multiply(const Tensor& a,bool transA,const Tensor& b,bool transB);
Tensor add(const Tensor& a,const Tensor& b);
void add_bias(Tensor& Y,const Tensor& b);
Tensor sum_rows(const Tensor& dY);
void add_bias_conv(Tensor& Yflat,const Tensor& b);
Tensor sum_spatial(const Tensor& dYflat);