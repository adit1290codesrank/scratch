#include "../../include/layer/res.h"

Res::~Res()
{
    for(auto layer:main) delete layer;
    for(auto layer:skip) delete layer;
}

void Res::add_main(Layer* layer){main.push_back(layer);}

void Res::add_skip(Layer* layer){skip.push_back(layer);}

Tensor Res::forward(const Tensor& X)
{
    Tensor F=X;
    for(auto layer:main) F=layer->forward(F);
    if(skip.empty()) return F+X;
    
    Tensor S=X;
    for(auto layer:skip) S=layer->forward(S);
    return F+S;
}

Tensor Res::backward(const Tensor& dY)
{
    Tensor dF=dY;
    for(int i=main.size()-1;i>=0;i--) dF=main[i]->backward(dF);
    if(skip.empty()) return dF+dY;

    Tensor dS=dY;
    for(int i=skip.size()-1;i>=0;i--)dS=skip[i]->backward(dS);
    return dF+dS;
}

std::vector<Tensor*> Res::get_weights()
{
    std::vector<Tensor*> weights;
    for(auto layer:main){auto l=layer->get_weights();weights.insert(weights.end(),l.begin(),l.end());}
    for(auto layer:skip){auto l=layer->get_weights();weights.insert(weights.end(),l.begin(),l.end());}
    return weights;
}

std::vector<Tensor*> Res::get_grads()
{
    std::vector<Tensor*> grads;
    for(auto layer:main){auto l=layer->get_grads();grads.insert(grads.end(),l.begin(),l.end());}
    for(auto layer:skip){auto l=layer->get_grads();grads.insert(grads.end(),l.begin(),l.end());}
    return grads;
}

std::vector<Tensor*> Res::get_states()
{
    std::vector<Tensor*> states;
    for(auto layer:main){auto l=layer->get_states();states.insert(states.end(),l.begin(),l.end());}
    for(auto layer:skip){auto l=layer->get_states();states.insert(states.end(),l.begin(),l.end());}
    return states;
}


