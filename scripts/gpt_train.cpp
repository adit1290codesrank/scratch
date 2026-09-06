#include "../include/core/gpt.h"
#include "../include/core/tensor.h"
#include "../include/core/optimizer.h"
#include "../include/core/loss.h"
#include <iostream>

int main()
{
    int vocab_size=50257,dmodel=768,heads=12,dff=3072,layers=12,seq_len=128,batch_size=2;

    std::cout<<"Building GPT-2 (124M Parameters)..."<<std::endl;
    GPT model(vocab_size,dmodel,heads,dff,layers,1024,0.1f);

    Adam optimizer(model.get_layers(),0.0003f);
    SparseCrossEntropyLoss loss_fn;
    
    model.compile(&optimizer,&loss_fn);

    std::cout<<"Model built successfully. Generating dummy batch..."<<std::endl;

    Tensor X=Tensor::zeros({batch_size,seq_len});
    Tensor Y=Tensor::zeros({batch_size,seq_len});
    Tensor mask=Tensor::ones({batch_size,seq_len});

    std::cout<<"Starting Training Loop..."<<std::endl;

    for(int epoch=1;epoch<=5;epoch++)
    {
        float loss=model.train_step(X,Y,&mask);
        std::cout<<"Step "<<epoch<<" | Loss: "<<loss<<std::endl;
    }

    std::cout<<"Training script executed successfully!"<<std::endl;
    return 0;
}
