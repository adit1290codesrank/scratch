#pragma once
#include <cstddef>

void* device_malloc(size_t bytes);
void device_free(void* ptr, size_t bytes);
void zero_malloc(void *ptr, size_t bytes);
void one_malloc(void *ptr, size_t bytes);
void copy_malloc(void *dest, const void *src, size_t bytes);
void randn_malloc(float *ptr,size_t count,float mean,float std);
void copy_from_host_malloc(void* dest, const void* src, size_t bytes);
void copy_to_host_malloc(void *dest,const void *src,size_t bytes);
void clear_memory_pool();
