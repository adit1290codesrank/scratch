#include "../../include/core/checkpoint.h"
#include "../../include/core/gpt.h"
#include "../../include/core/optimizer.h"
#include "../../include/core/data_loader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>

GPTDims read_gpt_dims(const std::string& path)
{
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path);
    char m[4]={0,0,0,0};
    f.read(m,4);
    if(m[0]!='G'||m[1]!='P'||m[2]!='T'||m[3]!='1') throw std::runtime_error(path+" is not a GPT weights file");
    int32_t h[6]={0,0,0,0,0,0};
    f.read((char*)h,sizeof(h));
    if(!f) throw std::runtime_error(path+" has a truncated header");
    return {h[0],h[1],h[2],h[3],h[4],h[5]};
}

void save_weights_only(const std::string& path,GPT& model)
{
    std::ofstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path+" for writing");
    model.save(f);
}

void load_weights_only(const std::string& path,GPT& model)
{
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path+" for reading");
    model.load(f);
}

void save_checkpoint(const std::string& path,GPT& model,Optimizer& opt,long long step,float best_loss,const DataLoader& loader)
{
    std::ofstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path+" for writing");

    const char magic[4]={'C','K','P','T'};
    f.write(magic,4);
    f.write((char*)&step,sizeof(step));
    f.write((char*)&best_loss,sizeof(best_loss));

    std::string rs=loader.rng_state();
    uint64_t rlen=rs.size();
    f.write((char*)&rlen,sizeof(rlen));
    f.write(rs.data(),(std::streamsize)rlen);

    model.save(f);
    opt.save_state(f);
}

void load_checkpoint(const std::string& path,GPT& model,Optimizer& opt,long long& step,float& best_loss,DataLoader& loader)
{
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path+" for reading");

    char magic[4]={0,0,0,0};
    f.read(magic,4);
    if(magic[0]!='C'||magic[1]!='K'||magic[2]!='P'||magic[3]!='T') throw std::runtime_error("bad checkpoint magic: "+path);

    f.read((char*)&step,sizeof(step));
    f.read((char*)&best_loss,sizeof(best_loss));

    uint64_t rlen=0;
    f.read((char*)&rlen,sizeof(rlen));
    if(!f||rlen>(1u<<20)) throw std::runtime_error("checkpoint corrupt: bad rng length");
    std::string rs(rlen,'\0');
    f.read(&rs[0],(std::streamsize)rlen);
    loader.set_rng_state(rs);

    model.load(f);
    opt.load_state(f);
}
