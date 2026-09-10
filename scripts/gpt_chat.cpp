#include "../include/core/gpt.h"
#include "../include/core/tokenizer.h"
#include "../include/core/checkpoint.h"
#include "../include/core/generator.h"
#include "../include/core/memory.h"
#include <iostream>
#include <string>
#include <vector>
#include <random>

int main(int argc,char** argv)
{
    std::string tok_path=argc>1?argv[1]:"data/tok.bin";
    std::string weights_path=argc>2?argv[2]:"data/model.bin";
    float temp=argc>3?std::stof(argv[3]):0.9f;
    int topk=argc>4?std::stoi(argv[4]):40;

    BPETokenizer tok;
    tok.load(tok_path);

    GPTDims d=read_gpt_dims(weights_path);
    if(d.vocab!=tok.size())
    {
        std::cerr<<"tokenizer vocab ("<<tok.size()<<") != model vocab ("<<d.vocab<<")\n";
        return 1;
    }

    GPT model(d.vocab,d.dmodel,d.heads,d.dff,d.layers,d.max_seq,0.0f);
    load_weights_only(weights_path,model);
    model.eval();

    std::cout<<"model "<<d.layers<<"L d"<<d.dmodel<<" | temp="<<temp<<" top_k="<<topk<<"\n";
    std::cout<<"chat ready (ctrl-D to quit, ':reset' to clear history)\n";

    std::random_device rd;
    std::vector<std::string> history;
    std::string line;
    while(true)
    {
        std::cout<<"you> "<<std::flush;
        if(!std::getline(std::cin,line)) break;
        if(line.empty()) continue;
        if(line==":reset"){history.clear();std::cout<<"[history cleared]\n";continue;}

        history.push_back(line);

        GenConfig gc;
        gc.temperature=temp;
        gc.top_k=topk;
        gc.max_new_tokens=200;
        gc.seed=rd();

        std::string reply=chat_generate(model,tok,history,d.max_seq,gc);
        history.push_back(reply);
        std::cout<<"bot> "<<reply<<"\n";
    }

    clear_memory_pool();
    return 0;
}
