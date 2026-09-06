#include "../../include/layer/decoder_gpt.h"
#include "../../include/core/tensor_ops.h"
#include "../../include/core/activation_ops.h"
#include "../../include/core/layernorm_ops.h"
#include <cmath>
#include <stdexcept>

DecoderGPT::DecoderGPT(int dmodel,int heads,int dff,int layers,float dr):dmodel(dmodel),heads(heads),Wq(dmodel,dmodel,Init::KAIMING,0.02f),Wk(dmodel,dmodel,Init::KAIMING,0.02f),Wv(dmodel,dmodel,Init::KAIMING,0.02f),Wo(dmodel,dmodel,Init::KAIMING,0.02f/sqrt(2.0f*layers)),FFN1(dmodel,dff,Init::KAIMING,0.02f),FFN2(dff,dmodel,Init::KAIMING,0.02f/sqrt(2.0f*layers)),attn_dropout(dr),ffn_dropout(dr)
{
    if(dmodel%heads!=0) throw std::invalid_argument("dmodel must be divisible by heads");
    this->dimension=dmodel/heads;
    g1=Tensor::ones({dmodel});g2=Tensor::ones({dmodel});
    b1=Tensor::zeros({dmodel});b2=Tensor::zeros({dmodel});
    dg1=Tensor::zeros({dmodel});dg2=Tensor::zeros({dmodel});db1=Tensor::zeros({dmodel});db2=Tensor::zeros({dmodel});
}

std::vector<Tensor*> DecoderGPT::get_weights()
{
    std::vector<Tensor*> weights;
    auto wq=Wq.get_weights();weights.insert(weights.end(),wq.begin(),wq.end());
    auto wk=Wk.get_weights();weights.insert(weights.end(),wk.begin(),wk.end());
    auto wv=Wv.get_weights();weights.insert(weights.end(),wv.begin(),wv.end());
    auto wo=Wo.get_weights();weights.insert(weights.end(),wo.begin(),wo.end());
    auto f1=FFN1.get_weights();weights.insert(weights.end(),f1.begin(),f1.end());
    auto f2=FFN2.get_weights();weights.insert(weights.end(),f2.begin(),f2.end());
    weights.push_back(&g1);weights.push_back(&b1);weights.push_back(&g2);weights.push_back(&b2);
    return weights;
}

std::vector<Tensor*> DecoderGPT::get_grads()
{
    std::vector<Tensor*> grads;
    auto wq=Wq.get_grads();grads.insert(grads.end(),wq.begin(),wq.end());
    auto wk=Wk.get_grads();grads.insert(grads.end(),wk.begin(),wk.end());
    auto wv=Wv.get_grads();grads.insert(grads.end(),wv.begin(),wv.end());
    auto wo=Wo.get_grads();grads.insert(grads.end(),wo.begin(),wo.end());
    auto f1=FFN1.get_grads();grads.insert(grads.end(),f1.begin(),f1.end());
    auto f2=FFN2.get_grads();grads.insert(grads.end(),f2.begin(),f2.end());
    grads.push_back(&dg1);grads.push_back(&db1);grads.push_back(&dg2);grads.push_back(&db2);
    return grads;
}

void DecoderGPT::train(){attn_dropout.train();ffn_dropout.train();}
void DecoderGPT::eval(){attn_dropout.eval();ffn_dropout.eval();}

Tensor DecoderGPT::forward(const Tensor& X,const Tensor* mask)
{
    this->cached_X=X;

    Tensor Y=Tensor::zeros(X.shape);
    layernorm_forward(X.get_data(),g1.get_data(),b1.get_data(),Y.get_data(),X.rows(),dmodel);

    this->cached_Q=Wq.forward(Y);
    this->cached_K=Wk.forward(Y);
    this->cached_V=Wv.forward(Y);

    masked_attention_forward(this->cached_Q,this->cached_K,this->cached_V,Y,heads,this->cached_S,mask);
    Y=Wo.forward(Y);
    Y=attn_dropout.forward(Y);

    Y=add(Y,X);

    this->cached_temp=Y;
    layernorm_forward(this->cached_temp.get_data(),g2.get_data(),b2.get_data(),Y.get_data(),Y.rows(),dmodel);

    Y=FFN1.forward(Y);
    relu_forward(Y);
    this->cached_relu=Y;
    Y=FFN2.forward(Y);
    Y=ffn_dropout.forward(Y);

    Y=add(Y,this->cached_temp);

    return Y;
}

Tensor DecoderGPT::backward(Tensor const& dY)
{
    Tensor dX=ffn_dropout.backward(dY);
    dX=FFN2.backward(dX);
    Tensor temp=Tensor::zeros(dX.shape);
    relu_backward(dX,this->cached_relu,temp);
    dX=FFN1.backward(temp);
  
    layernorm_backward(dX.get_data(),cached_temp.get_data(),g2.get_data(),dg2.get_data(),db2.get_data(),temp.get_data(),dY.rows(),dmodel);
    dX=add(dY,temp);
  
    Tensor dtemp=dX;
  
    dX=attn_dropout.backward(dX);
    dX=Wo.backward(dX);
  
    Tensor dQ=Tensor::zeros(cached_Q.shape),dK=Tensor::zeros(cached_K.shape),dV=Tensor::zeros(cached_V.shape);
    masked_attention_backward(dX,cached_Q,cached_K,cached_V,cached_S,dQ,dK,dV,heads);
  
    dQ=Wq.backward(dQ);dK=Wk.backward(dK);dV=Wv.backward(dV);
    dX=add(add(dQ,dK),dV);
  
    layernorm_backward(dX.get_data(),cached_X.get_data(),g1.get_data(),dg1.get_data(),db1.get_data(),temp.get_data(),dY.rows(),dmodel);
    dX=add(dtemp,temp);
  
    return dX;
}
