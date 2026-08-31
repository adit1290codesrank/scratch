#include "../../include/core/optimizer.h"
#include "../../include/core/optimizers_ops.h"

Adam::Adam(std::vector<Layer*> layers,float lr,float b1,float b2,float e):Optimizer(layers,lr),b1(b1),b2(b2),e(e),t(0)
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
        for(int i=0;i<n;i++) adam(weights[i],grads[i],m_map[weights[i]],v_map[weights[i]],lr,b1,b2,b1t,b2t,e);
    }
}