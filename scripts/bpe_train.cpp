#include "../include/core/tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc,char** argv)
{
    if(argc<4)
    {
        std::cerr<<"usage: "<<argv[0]<<" <out.bin> <vocab_size> <corpus1> [corpus2 ...]\n";
        return 1;
    }
    std::string out=argv[1];
    int vocab=std::stoi(argv[2]);
    std::vector<std::string> paths;
    for(int i=3;i<argc;i++) paths.push_back(argv[i]);

    BPETokenizer tk;
    tk.train(paths,vocab);
    tk.save(out);
    std::cout<<"saved "<<out<<" | vocab_size="<<tk.size()<<"\n";
    return 0;
}
