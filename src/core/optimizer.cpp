#include "../../include/core/optimizer.h"
#include "../../include/core/optimizers_ops.h"
#include <cmath>
#include <ostream>
#include <istream>
#include <vector>
#include <stdexcept>

Adam::Adam(std::vector<Layer*> layers,float lr,float b1,float b2,float e,float wd):Optimizer(layers,lr),b1(b1),b2(b2),e(e),wd(wd),t(0)
{
    for(Layer* layer:layers)
    {
        auto weights=layer->get_weights();
        for(auto weight:weights)
        {
            m_map[weight]=Tensor::zeros(weight->shape);
            v_map[weight]=Tensor::zeros(weight->shape);
        }
    }
}

void Adam::step()
{
    this->t++;
    float b1t=pow(b1,t),b2t=pow(b2,t);
    for(Layer* layer:layers)
    {
        auto weights=layer->get_weights(),grads=layer->get_grads();
        int n=weights.size();
        for(int i=0;i<n;i++) adam(weights[i],grads[i],m_map[weights[i]],v_map[weights[i]],lr,b1,b2,b1t,b2t,e,wd);
    }
}

void Adam::save_state(std::ostream& os)
{
    os.write((char*)&t,sizeof(t));
    std::vector<float> buf;
    for(Layer* layer:layers)
        for(Tensor* w:layer->get_weights())
        {
            Tensor& m=m_map[w];Tensor& v=v_map[w];
            buf.resize(m.total_elements());
            m.copy_to_host(buf.data());
            os.write((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
            v.copy_to_host(buf.data());
            os.write((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
        }
}

void Adam::load_state(std::istream& is)
{
    is.read((char*)&t,sizeof(t));
    std::vector<float> buf;
    for(Layer* layer:layers)
        for(Tensor* w:layer->get_weights())
        {
            Tensor& m=m_map[w];Tensor& v=v_map[w];
            buf.resize(m.total_elements());
            is.read((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
            if(!is) throw std::runtime_error("optimizer state truncated");
            m.copy_from_host(buf.data());
            is.read((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
            if(!is) throw std::runtime_error("optimizer state truncated");
            v.copy_from_host(buf.data());
        }
}