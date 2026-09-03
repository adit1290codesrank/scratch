#pragma once

void dropout_forward_gpu(const float* X,float* Y,float* mask,float p,int size,unsigned int seed);
void dropout_backward_gpu(const float* dY,float* dX,const float* mask,float p,int size);
