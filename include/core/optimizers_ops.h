#pragma once
#include "tensor.h"

void adam(Tensor *W,Tensor *dW,Tensor& m,Tensor& v,float lr,float b1,float b2,float b1t,float b2t,float e,float wd);