#include "../../include/core/gpt.h"
#include "../../include/core/activation_ops.h"
#include <cuda_runtime.h> // For softmax during inference if needed

GPT::GPT(int vocab_size,int dmodel,int heads,int dff,int layers,int max_seq,float dr):vocab_size(vocab_size),dmodel(dmodel),heads(heads),layers(layers),max_seq(max_seq),embed(vocab_size,dmodel,max_seq),final_norm(dmodel),lm_head(dmodel,vocab_size)
{
    for(int i=0;i<layers;i++)blocks.push_back(std::make_unique<DecoderGPT>(dmodel,heads,dff,layers,dr));
}

void GPT::compile(Optimizer* opt,Loss* loss)
{
    this->optimizer=opt;
    this->loss=loss;
}

Tensor GPT::forward(const Tensor& X,const Tensor* pad_mask)
{
    Tensor Y=embed.forward(X,pad_mask);
    for(auto& block:blocks)Y=block->forward(Y,pad_mask);
    Y=final_norm.forward(Y,pad_mask);
    Y=lm_head.forward(Y,pad_mask);
    softmax_forward(Y); 
    return Y;
}

void GPT::backward(const Tensor& dY)
{
    Tensor dX=lm_head.backward(dY);
    dX=final_norm.backward(dX);
    for(int i=blocks.size()-1;i>=0;i--)dX=blocks[i]->backward(dX);
    embed.backward(dX);
}

std::vector<Tensor*> GPT::get_all_weights()
{
    std::vector<Tensor*> weights;
    auto w=embed.get_weights(); weights.insert(weights.end(),w.begin(),w.end());
    for(auto& block:blocks)
    {
        auto bw=block->get_weights();
        weights.insert(weights.end(),bw.begin(),bw.end());
    }
    auto fn=final_norm.get_weights(); weights.insert(weights.end(),fn.begin(),fn.end());
    auto lm=lm_head.get_weights(); weights.insert(weights.end(),lm.begin(),lm.end());
    return weights;
}

std::vector<Tensor*> GPT::get_all_grads()
{
    std::vector<Tensor*> grads;
    auto g=embed.get_grads(); grads.insert(grads.end(),g.begin(),g.end());
    for(auto& block:blocks)
    {
        auto bg=block->get_grads();
        grads.insert(grads.end(),bg.begin(),bg.end());
    }
    auto fn=final_norm.get_grads(); grads.insert(grads.end(),fn.begin(),fn.end());
    auto lm=lm_head.get_grads(); grads.insert(grads.end(),lm.begin(),lm.end());
    return grads;
}


std::vector<Layer*> GPT::get_layers()
{
    std::vector<Layer*> l;
    l.push_back(&embed);
    for(auto& b:blocks) l.push_back(b.get());
    l.push_back(&final_norm);
    l.push_back(&lm_head);
    return l;
}

float GPT::train_step(const Tensor& X,const Tensor& targets,const Tensor* pad_mask)
{
    for(auto grad:get_all_grads()) cudaMemset(grad->get_data(),0,grad->total_elements()*sizeof(float));

    Tensor probs=forward(X,pad_mask);
    float loss_val=loss->calculate_loss(probs,targets);
    Tensor dY=loss->backward_loss(probs,targets);
    
    backward(dY);
    optimizer->step();
    
    return loss_val;
}
