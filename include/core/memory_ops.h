#pragma once
#include <cstddef>

float* raw_device_malloc(size_t bytes);
void raw_device_free(float* ptr);
void raw_zero_malloc(float* ptr, size_t bytes);
void raw_copy_malloc(float* dest, float* src, size_t bytes);
void raw_randn(float* ptr,size_t size,float mean,float std);
