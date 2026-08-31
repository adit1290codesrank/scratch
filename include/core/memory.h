#pragma once
#include <cstddef>

float* device_malloc(size_t bytes);
void device_free(float* ptr, size_t bytes);
void zero_malloc(float *ptr, size_t bytes);
void copy_malloc(float *dest, float *src, size_t bytes);
void randn_malloc(float *ptr,size_t bytes,float mean,float std);
