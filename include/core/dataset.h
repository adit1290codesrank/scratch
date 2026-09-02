#pragma once
#include <string>
#include "tensor.h"

class Dataset
{
    public:
        static void load_mnist(const std::string& path,const std::string& label,Tensor& X,Tensor& Y);
        static void load_emnist(const std::string& path,const std::string& label,Tensor& X,Tensor& Y,int nc);
};