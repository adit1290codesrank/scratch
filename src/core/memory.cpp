#include "../../include/core/memory_ops.h"
#include <unordered_map>
#include <vector>
#include <mutex>

static std::unordered_map<size_t,std::vector<float*>> memory_pool;
static std::mutex pool_mutex;

float* device_malloc(size_t bytes) 
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (memory_pool.find(bytes)!=memory_pool.end() && !memory_pool[bytes].empty()) 
    {
        float* ptr=memory_pool[bytes].back();
        memory_pool[bytes].pop_back();
        return ptr;
    }
    
    return raw_device_malloc(bytes);
}

void device_free(float* ptr, size_t bytes) 
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    memory_pool[bytes].push_back(ptr);
}

void zero_malloc(float *ptr, size_t bytes) {raw_zero_malloc(ptr, bytes);}

void copy_malloc(float *dest, float *src, size_t bytes) {raw_copy_malloc(dest, src, bytes);}

void randn_malloc(float *ptr,size_t bytes,float mean,float std){raw_randn(ptr,bytes,mean,std);}