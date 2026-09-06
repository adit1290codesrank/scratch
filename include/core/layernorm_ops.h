#pragma once
#include "tensor.h"

void layernorm_forward(const float* X,const float* g,const float* b,float* Y,int n,int d,float eps=1e-5f);
void layernorm_backward(const float* dY,const float* X,const float* g,float* dg,float* db,float* dX,int n,int d,float eps=1e-5f);
void layernorm_backward(const float* dY,const float* X,const float* gamma,float* dgamma,float* dbeta,float* dX,int N,int D,float eps=1e-5f);
