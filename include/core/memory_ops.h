#pragma once
#include <cstddef>

void* raw_device_malloc(size_t bytes);
void raw_device_free(void* ptr);
void raw_zero_malloc(void* ptr, size_t bytes);
void raw_one_malloc(void* ptr, size_t bytes);
void raw_copy_malloc(void* dest, const void* src, size_t bytes);
void raw_randn(float* ptr,size_t count,float mean,float std);
void raw_copy_from_host(void* dest, const void* src, size_t bytes);
void raw_copy_to_host(void* dest, const void* src, size_t bytes);
