#pragma once
#include "tensor.h"

void im2col_gpu(const float* im,int c,int h,int w,int k,int p,int s,float* col);
void col2im_gpu(const float* col,int c,int h,int w,int k,int p,int s,float* im);