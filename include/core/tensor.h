#pragma once
#include <memory>
#include <vector>

enum class DType { F32, BF16 };

class Tensor
{
    private:
        std::shared_ptr<void> data;
        DType dtype_=DType::F32;

    public:
        std::vector<int> shape;

        Tensor():shape({0}){}
        int rows() const {return shape.empty()?0:shape[0];}
        int cols() const
        {
            if(shape.size()<2) return 1;
            int total=1;
            for(int i=1;i<shape.size();i++) total*=shape[i];
            return total;
        }
        size_t total_elements() const {return rows()*cols();}

        DType dtype() const {return dtype_;}
        size_t elem_size() const {return dtype_==DType::F32?sizeof(float):2;}
        size_t total_bytes() const {return total_elements()*elem_size();}

        Tensor(std::vector<int> shape, DType dt=DType::F32);

        float *get_data() const {return (float*)data.get();}

        Tensor reshape(std::vector<int> shape) const;
        static Tensor zeros(std::vector<int> shape);//belongs to class not object
        static Tensor ones(std::vector<int> shape);
        static Tensor randn(std::vector<int> shape,float mean,float std);
        Tensor operator*(const Tensor& other) const;
        Tensor operator+(const Tensor& other) const;
        Tensor clone() const;
        Tensor slice(int start,int end) const;
        void copy_from_host(const float* host_data) const;
        void copy_to_host(float* host_data) const;
};

