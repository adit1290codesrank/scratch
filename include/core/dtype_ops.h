#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float to_float(float x){return x;}
__device__ __forceinline__ float to_float(__nv_bfloat16 x){return __bfloat162float(x);}
template<typename T> __device__ __forceinline__ T from_float(float x);
template<> __device__ __forceinline__ float from_float<float>(float x){return x;}
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x){return __float2bfloat16(x);}