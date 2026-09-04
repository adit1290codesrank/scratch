#pragma once
#include "tensor.h"

float mse_forward(const Tensor& pred, const Tensor& target);
void mse_backward(const Tensor& pred, const Tensor& target, Tensor& dY);

float ce_forward(const Tensor& pred, const Tensor& target);
void ce_backward(const Tensor& pred, const Tensor& target, Tensor& dY);

float ls_ce_forward(const Tensor& pred, const Tensor& target,int n,float a);
void ls_ce_backward(const Tensor& pred, const Tensor& target, Tensor& dY,int n,float a);