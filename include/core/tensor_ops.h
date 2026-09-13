#pragma once
#include "tensor.h"

Tensor multiply(const Tensor& a,bool transA,const Tensor& b,bool transB);
Tensor add(const Tensor& a,const Tensor& b);
void cast_tensor(Tensor& dst,const Tensor& src);
void add_bias(Tensor& Y,const Tensor& b);
Tensor sum_rows(const Tensor& dY);
void add_bias_conv(Tensor& Yflat,const Tensor& b);
Tensor sum_spatial(const Tensor& dYflat);

Tensor multiply_conv_forward(const Tensor& W,const Tensor& Xcol);
Tensor multiply_conv_backward_dX(const Tensor& W,const Tensor& dYflat);
void multiply_conv_backward_dW(Tensor& dW,const Tensor& dYflat,const Tensor& Xcol);

void attention_forward(const Tensor& Q,const Tensor& K, const Tensor& V,Tensor& Z,int heads,Tensor& S,const Tensor* mask=nullptr);
void attention_backward(Tensor& dZ,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads);

void masked_attention_forward(const Tensor& Q,const Tensor& K,const Tensor& V,Tensor& Z,int heads,Tensor& S,const Tensor* mask=nullptr);
void masked_attention_backward(Tensor& dZ,Tensor& Q,Tensor& K,Tensor& V,Tensor& S,Tensor& dQ,Tensor& dK,Tensor& dV,int heads);

void embedding_forward(const Tensor& X,const Tensor& W_tok,const Tensor& W_pos,Tensor& Y,int batch,int seq_len,int dmodel);
void embedding_backward(const Tensor& X,const Tensor& dY,Tensor& dW_tok,Tensor& dW_pos,int batch,int seq_len,int dmodel);
