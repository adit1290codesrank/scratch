#include "../../include/core/gpt.h"
#include "../../include/core/activation_ops.h"
#include <cuda_runtime.h> // For softmax during inference if needed
#include <fstream>
#include <stdexcept>
#include <cstdint>

GPT::GPT(int vocab_size,int dmodel,int heads,int dff,int layers,int max_seq,float dr):vocab_size(vocab_size),dmodel(dmodel),heads(heads),dff(dff),layers(layers),max_seq(max_seq),embed(vocab_size,dmodel,max_seq),final_norm(dmodel),lm_head(dmodel,vocab_size)
{
    for(int i=0;i<layers;i++)blocks.push_back(std::make_unique<DecoderGPT>(dmodel,heads,dff,layers,dr));
}

void GPT::compile(Optimizer* opt,Loss* loss)
{
    this->optimizer=opt;
    this->loss=loss;
}

Tensor GPT::logits(const Tensor& X,const Tensor* pad_mask)
{
    Tensor Y=embed.forward(X,pad_mask);
    for(auto& block:blocks)Y=block->forward(Y,pad_mask);
    Y=final_norm.forward(Y,pad_mask);
    Y=lm_head.forward(Y,pad_mask);
    return Y;
}

Tensor GPT::forward(const Tensor& X,const Tensor* pad_mask)
{
    Tensor Y=logits(X,pad_mask);
    softmax_forward(Y);
    return Y;
}

void GPT::train(){for(Layer* L:get_layers())L->train();}
void GPT::eval(){for(Layer* L:get_layers())L->eval();}

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

void GPT::save(std::ostream& f)
{
    const char magic[4]={'G','P','T','1'};
    f.write(magic,4);
    int32_t hdr[6]={vocab_size,dmodel,heads,dff,layers,max_seq};
    f.write((char*)hdr,sizeof(hdr));

    std::vector<float> buf;
    for(Layer* L:get_layers())
    {
        for(Tensor* w:L->get_weights())
        {
            buf.resize(w->total_elements());
            w->copy_to_host(buf.data());
            f.write((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
        }
        for(Tensor* s:L->get_states())
        {
            buf.resize(s->total_elements());
            s->copy_to_host(buf.data());
            f.write((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
        }
    }
}

void GPT::load(std::istream& f)
{
    char magic[4]={0,0,0,0};
    f.read(magic,4);
    if(magic[0]!='G'||magic[1]!='P'||magic[2]!='T'||magic[3]!='1') throw std::runtime_error("bad GPT checkpoint magic");

    int32_t hdr[6]={0,0,0,0,0,0};
    f.read((char*)hdr,sizeof(hdr));
    if(hdr[0]!=vocab_size||hdr[1]!=dmodel||hdr[2]!=heads||hdr[3]!=dff||hdr[4]!=layers||hdr[5]!=max_seq)
        throw std::runtime_error("GPT checkpoint architecture mismatch");

    std::vector<float> buf;
    for(Layer* L:get_layers())
    {
        for(Tensor* w:L->get_weights())
        {
            buf.resize(w->total_elements());
            f.read((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
            if(!f) throw std::runtime_error("GPT checkpoint truncated");
            w->copy_from_host(buf.data());
        }
        for(Tensor* s:L->get_states())
        {
            buf.resize(s->total_elements());
            f.read((char*)buf.data(),(std::streamsize)(buf.size()*sizeof(float)));
            if(!f) throw std::runtime_error("GPT checkpoint truncated");
            s->copy_from_host(buf.data());
        }
    }
}
