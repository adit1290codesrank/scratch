#include "../../include/core/memory_ops.h"
#include <unordered_map>
#include <vector>
#include <mutex>

static std::unordered_map<size_t,std::vector<void*>> memory_pool;
static std::mutex pool_mutex;

void* device_malloc(size_t bytes)
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (memory_pool.find(bytes)!=memory_pool.end() && !memory_pool[bytes].empty())
    {
        void* ptr=memory_pool[bytes].back();
        memory_pool[bytes].pop_back();
        return ptr;
    }

    return raw_device_malloc(bytes);
}

void device_free(void* ptr, size_t bytes)
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    memory_pool[bytes].push_back(ptr);
}

void zero_malloc(void *ptr, size_t bytes) {raw_zero_malloc(ptr, bytes);}

void one_malloc(void *ptr, size_t count, DType dt) {raw_one_malloc(ptr, count, dt);}

void copy_malloc(void *dest, const void *src, size_t bytes) {raw_copy_malloc(dest, src, bytes);}

void randn_malloc(float *ptr,size_t count,float mean,float std){raw_randn(ptr,count,mean,std);}

void copy_from_host_malloc(void* dest, const void* src, size_t bytes){raw_copy_from_host(dest,src,bytes);}

void copy_to_host_malloc(void *dest,const void *src,size_t bytes) {raw_copy_to_host(dest, src, bytes);}

void clear_memory_pool()
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    for(auto& pair:memory_pool) for(void* ptr:pair.second) raw_device_free(ptr);
    memory_pool.clear();
}
