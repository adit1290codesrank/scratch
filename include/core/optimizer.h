#pragma once
#include <vector>
#include <unordered_map>
#include "../layer/layer.h"

class Optimizer
{
    protected:
        std::vector<Layer*> layers;
        float lr;
    
    public:
        Optimizer(std::vector<Layer*> layers,float lr):layers(layers),lr(lr){}
        virtual ~Optimizer()=default;

        virtual void step() = 0;
};

class Adam:public Optimizer
{
    private:
        float b1,b2,e;
        int t;

        std::unordered_map<Tensor*,Tensor> m_map;
        std::unordered_map<Tensor*,Tensor> v_map;

    public:

        Adam(std::vector<Layer*> layers,float lr=0.001f,float beta1=0.9f,float beta2=0.999f,float eps=1e-8f);
        void step() override;
};