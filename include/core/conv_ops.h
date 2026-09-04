#pragma once
#include "tensor.h"

void im2col_gpu(const float* im,int batch,int c,int h,int w,int k,int p,int s,float* col);
void col2im_gpu(const float* col,int batch,int c,int h,int w,int k,int p,int s,float* im);
void maxpool_forward_gpu(const float* X,float* Y,float* mask,int b,int c,int h,int w,int hout,int wout,int k,int s);
void maxpool_backward_gpu(const float* dY,const float* mask,float* dX,int b,int c,int h,int w,int hout, int wout);
