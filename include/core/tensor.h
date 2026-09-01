#pragma once
#include <memory>
#include <vector>

class Tensor
{
    private:
        std::shared_ptr<float> data;

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

        Tensor(std::vector<int> shape);

        float *get_data() const {return data.get();}

        Tensor reshape(std::vector<int> shape) const;
        static Tensor zeros(std::vector<int> shape);//belongs to class not object
        static Tensor randn(std::vector<int> shape,float mean,float std);
        Tensor operator*(const Tensor& other) const;
        Tensor operator+(const Tensor& other) const;
        Tensor clone() const;
        Tensor slice(int start,int end) const;
        void copy_from_host(const float* host_data) const;
        void copy_to_host(float* host_data) const;
};

