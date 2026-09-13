#include "../../include/core/tensor.h"
#include "../../include/core/memory.h"
#include "../../include/core/tensor_ops.h"
#include <stdexcept>

Tensor::Tensor(std::vector<int> shape, DType dt):dtype_(dt),shape(shape)
{
    size_t bytes=this->total_bytes();
    void* ptr = device_malloc(bytes);
    this->data = std::shared_ptr<void>(ptr, [bytes](void* p) {device_free(p, bytes);});
}

Tensor Tensor::reshape(std::vector<int> new_shape) const
{
    size_t new_total=1;
    for(int i:new_shape)new_total*=i;

    if(new_total!=this->total_elements()) throw std::invalid_argument("Total elements must remain the same in reshape");
    Tensor temp=*this;
    temp.shape=new_shape;
    return temp;
}

Tensor Tensor::zeros(std::vector<int> shape, DType dt)
{
    Tensor temp(shape,dt);
    zero_malloc(temp.get_data(),temp.total_bytes());
    return temp;
}

Tensor Tensor::ones(std::vector<int> shape, DType dt)
{
    Tensor temp(shape,dt);
    one_malloc(temp.get_data(),temp.total_elements(),dt);
    return temp;
}

Tensor Tensor::randn(std::vector<int> shape,float mean,float std)
{
    Tensor temp(shape);
    randn_malloc(temp.get_data(),temp.total_elements(),mean,std);
    return temp;
}

Tensor Tensor::clone() const
{
    Tensor temp(this->shape,this->dtype_);
    copy_malloc(temp.get_data(),this->get_data(),this->total_bytes());
    return temp;
}

void Tensor::copy_from_host(const float* host_data) const
{
    if(dtype_==DType::F32)
    {
        copy_from_host_malloc(this->get_data(),host_data,this->total_bytes());
    }
    else
    {
        Tensor tmp(shape,DType::F32);
        tmp.copy_from_host(host_data);
        cast_tensor(const_cast<Tensor&>(*this),tmp);
    }
}

void Tensor::copy_to_host(float* host_data) const
{
    if(dtype_==DType::F32)
    {
        copy_to_host_malloc(host_data,this->get_data(),this->total_bytes());
    }
    else
    {
        Tensor tmp(shape,DType::F32);
        cast_tensor(tmp,*this);
        tmp.copy_to_host(host_data);
    }
}

Tensor Tensor::operator*(const Tensor& other) const{return multiply(*this,false,other,false);};

Tensor Tensor::operator+(const Tensor& other) const{return add(*this,other);};

Tensor Tensor::slice(int start,int end) const
{
    std::vector<int> new_shape=shape;
    new_shape[0]=end-start;

    Tensor temp=Tensor::zeros(new_shape);
    size_t offset_bytes=(size_t)start*(total_elements()/shape[0])*elem_size();

    copy_malloc(temp.get_data(),(char*)get_data()+offset_bytes,temp.total_bytes());
    return temp;
}
