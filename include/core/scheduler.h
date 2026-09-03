#pragma once
#include "optimizer.h"

class Scheduler
{
    protected:
        Optimizer* optimizer;

    public:
        Scheduler(Optimizer* optimizer):optimizer(optimizer){}
        virtual ~Scheduler()=default;
        virtual void step()=0;
};

class CosineAnnealing:public Scheduler
{
    private:
        float lr_max,lr_min;
        int total,current;

    public:
        CosineAnnealing(Optimizer* optimizer,float lr_max,float lr_min,int total);
        void step() override;
};