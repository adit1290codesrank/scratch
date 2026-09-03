#pragma once

struct AugmentParams{int flip,dx,dy,cx,cy,hole;};

void* alloc_augment(int b);
void free_augment(void* ptr);

void augment_gpu(const float* X,float* Y,void* pgpu,const AugmentParams* p,int b,int c,int h,int w);
