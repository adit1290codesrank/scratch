#pragma once
#include "tensor.h"
#include <vector>

void zero_grads(const std::vector<Tensor*>& grads);
float clip_grad_norm(const std::vector<Tensor*>& grads, float max_norm);