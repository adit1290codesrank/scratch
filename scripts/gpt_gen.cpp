#include "../include/core/gpt.h"
#include "../include/core/tokenizer.h"
#include "../include/core/checkpoint.h"
#include "../include/core/generator.h"
#include "../include/core/memory.h"
#include <iostream>
#include <string>
#include <random>

int main(int argc,char** argv)
{
    std::string tok_path=argc>1?argv[1]:"data/tok.bin";
    std::string weights_path=argc>2?argv[2]:"data/model.bin";
    float temp=argc>3?std::stof(argv[3]):0.8f;
    int topk=argc>4?std::stoi(argv[4]):40;
    int n=argc>5?std::stoi(argv[5]):200;

    BPETokenizer tok;
    tok.load(tok_path);

    GPTDims d=read_gpt_dims(weights_path);
    GPT model(d.vocab,d.dmodel,d.heads,d.dff,d.layers,d.max_seq,0.0f);
    load_weights_only(weights_path,model);
    model.eval();

    int EOT=tok.token_to_id("<|endoftext|>");
    std::random_device rd;
    std::string line;
    std::cout<<"prompt> "<<std::flush;
    while(std::getline(std::cin,line))
    {
        std::vector<int> ids=tok.encode(line);
        GenConfig gc;gc.temperature=temp;gc.top_k=topk;gc.max_new_tokens=n;gc.seed=rd();
        std::vector<int> out=generate(model,ids,d.max_seq,gc,EOT);
        std::cout<<line<<tok.decode(out)<<"\n\nprompt> "<<std::flush;
    }
    clear_memory_pool();
    return 0;
}