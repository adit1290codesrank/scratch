#pragma once

void bn_param_gpu(const float* X,float* mean,float* var,int n,int c,int s);
void bn_forward_gpu(const float* X,const float* mean,const float* var,const float* g,const float* b,float* Y,float* X_,int n,int c,int s,float e);
void bn_running_gpu(float *rm,float *rv,const float *bm,const float* bv,float m,int c);
void bn_backwards_param_gpu(const float* dY,const float* X_,float* dg,float* db,int n,int c,int s);
void bn_backward_gpu(const float* dY,const float* X_,const float* var,const float* g,float* dX,float* dg,float* db,int n,int c,int s,float e);
