#pragma once
#include "tensor.h"
#include "loss.h"
#include "optimizer.h"
#include "../layer/embedding.h"
#include "../layer/decoder_gpt.h"
#include "../layer/layernorm.h"
#include "../layer/linear.h"
#include <vector>
#include <memory>

class GPT
{
    private:
        int vocab_size,dmodel,heads,layers,max_seq;
        Embedding embed;
        std::vector<std::unique_ptr<DecoderGPT>> blocks;
        LayerNorm final_norm;
        Linear lm_head;
        Optimizer* optimizer;
        Loss* loss;

    public:
        GPT(int vocab_size,int dmodel,int heads,int dff,int layers,int max_seq=1024,float dr=0.1f);
        
        void compile(Optimizer* opt,Loss* loss);
        Tensor forward(const Tensor& X,const Tensor* mask=nullptr);
        void backward(const Tensor& dY);
        float train_step(const Tensor& X,const Tensor& targets,const Tensor* mask=nullptr);
        
        std::vector<Tensor*> get_all_weights();
        std::vector<Tensor*> get_all_grads();
        std::vector<Layer*> get_layers();
};
