#include "../../include/core/tensor.h"
#include "../../include/core/memory.h"
#include "../../include/core/tensor_ops.h"
#include <stdexcept>

Tensor::Tensor(std::vector<int> shape):shape(shape)
{
    size_t total=this->total_elements();
    size_t bytes=total*sizeof(float);
    float* ptr = device_malloc(bytes);
    this->data = std::shared_ptr<float>(ptr, [bytes](float* p) {device_free(p, bytes);});
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

Tensor Tensor::zeros(std::vector<int> shape)
{
    Tensor temp(shape);
    size_t bytes=temp.total_elements()*sizeof(float);
    zero_malloc(temp.get_data(),bytes);
    return temp;
}

Tensor Tensor::ones(std::vector<int> shape)
{
    Tensor temp(shape);
    size_t bytes=temp.total_elements()*sizeof(float);
    one_malloc(temp.get_data(),bytes);
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
    Tensor temp(this->shape);
    size_t bytes=this->total_elements()*sizeof(float);
    copy_malloc(temp.get_data(),this->get_data(),bytes);
    return temp;
}

void Tensor::copy_from_host(const float* host_data) const
{
    size_t bytes=this->total_elements()*sizeof(float);
    copy_from_host_malloc(this->get_data(),host_data,bytes);
}

void Tensor::copy_to_host(float* host_data) const
{
    size_t bytes=this->total_elements()*sizeof(float);
    copy_to_host_malloc(host_data,this->get_data(),bytes);
}

Tensor Tensor::operator*(const Tensor& other) const{return multiply(*this,false,other,false);};

Tensor Tensor::operator+(const Tensor& other) const{return add(*this,other);};

Tensor Tensor::slice(int start,int end) const
{
    std::vector<int> new_shape=shape;
    new_shape[0]=end-start;
    
    Tensor temp=Tensor::zeros(new_shape);
    size_t bytes=temp.total_elements()*sizeof(float);
    size_t offset=start*(total_elements()/shape[0]);
    
    copy_malloc(temp.get_data(),get_data()+offset,bytes);
    return temp;
}