#include "../../include/core/scheduler.h"
#include <cmath>

CosineAnnealing::CosineAnnealing(Optimizer* optimizer,float lr_max,float lr_min,int total):Scheduler(optimizer),lr_max(lr_max),lr_min(lr_min),total(total),current(0){}

void CosineAnnealing::step()
{
    int t=(total>1)?(total-1):1;
    float prog=(float)current/t;

    if(prog>1.0f) prog=1.0f;
    float new_lr=lr_min+0.5f*(lr_max-lr_min)*(1.0f+std::cos(3.14159265f*prog));
    optimizer->set_lr(new_lr);
    current++;
}