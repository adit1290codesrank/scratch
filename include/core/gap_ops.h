#pragma once

void gap_forward_gpu(const float* X,float* Y,int n,int c,int s);
void gap_backward_gpu(const float* dY,float* dX,int n,int c,int s);
