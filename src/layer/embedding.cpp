#include "../../include/layer/embedding.h"
#include "../../include/core/tensor_ops.h"
#include <cmath>
#include <stdexcept>

Embedding::Embedding(int vocab_size,int dmodel,int max_seq):vocab_size(vocab_size),dmodel(dmodel),max_seq(max_seq)
{
    W_token=Tensor::randn({vocab_size,dmodel},0.0f,0.02f);
    W_pos=Tensor::randn({max_seq,dmodel},0.0f,0.02f);
    dW_token=Tensor::zeros({vocab_size,dmodel});
    dW_pos=Tensor::zeros({max_seq,dmodel});
}

std::vector<Tensor*> Embedding::get_weights(){return {&W_token,&W_pos};}

std::vector<Tensor*> Embedding::get_grads(){return {&dW_token,&dW_pos};}

Tensor Embedding::forward(const Tensor& X,const Tensor* pad_mask)
{
    this->cached_X=X;
    int batch=X.rows(),seq_len=X.cols();
    if(seq_len>max_seq) throw std::invalid_argument("Sequence length exceeds max_seq");

    Tensor Y=Tensor::zeros({batch,seq_len,dmodel});
    embedding_forward(X,W_token,W_pos,Y,batch,seq_len,dmodel);
    return Y;
}

Tensor Embedding::backward(const Tensor& dY)
{
    int batch=cached_X.rows(),seq_len=cached_X.cols();
    embedding_backward(cached_X,dY,dW_token,dW_pos,batch,seq_len,dmodel);
    return Tensor::zeros(cached_X.shape); 
}
