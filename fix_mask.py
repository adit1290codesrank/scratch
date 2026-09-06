import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the signature
    content = content.replace("Tensor forward(const Tensor& X,const Tensor* mask)", "Tensor forward(const Tensor& X,const Tensor* pad_mask)")
    content = content.replace("Tensor forward(const Tensor& X)", "Tensor forward(const Tensor& X,const Tensor* pad_mask)")
    
    # In encoder.cpp and decoder_gpt.cpp
    if "encoder.cpp" in filepath or "decoder_gpt.cpp" in filepath:
        content = content.replace("attention_forward(", "attention_forward(") # just to be safe
        content = content.replace("cached_attn,mask);", "cached_attn,pad_mask);")
        content = content.replace("cached_S,mask);", "cached_S,pad_mask);")
    
    # In gpt.cpp
    if "gpt.cpp" in filepath:
        content = content.replace("forward(X,mask)", "forward(X,pad_mask)")
        content = content.replace("forward(Y,mask)", "forward(Y,pad_mask)")
        content = content.replace("train_step(const Tensor& X,const Tensor& targets,const Tensor* mask)", "train_step(const Tensor& X,const Tensor& targets,const Tensor* pad_mask)")
        
    with open(filepath, 'w') as f:
        f.write(content)

fix_file("src/layer/encoder.cpp")
fix_file("src/core/gpt.cpp")
