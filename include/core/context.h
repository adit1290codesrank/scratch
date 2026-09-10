#pragma once
#include <cublas_v2.h>
#include <stdexcept>

class Context
{
    private:
        cublasHandle_t handle;
        Context()
        {
            if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS initialization failed!");
            cublasSetMathMode(handle,CUBLAS_TF32_TENSOR_OP_MATH);
        }
        ~Context(){cublasDestroy(handle);}

    public:
        Context(const Context&)=delete;
        Context& operator=(const Context&)=delete;

        static Context& get_instance()
        {
            static Context instance;
            return instance;
        }

        cublasHandle_t get_cublas_handle() const
        {
            return handle;
        }
};