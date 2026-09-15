#include "../../include/core/data_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

DataLoader::DataLoader(BPETokenizer& t,int block,int batch,uint64_t seed)
    :tok(t),block(block),batch(batch),rng(seed),hX(batch*block),hY(batch*block)
{
    if(block<1||batch<1) throw std::invalid_argument("block and batch must be >= 1");
    if(tok.size()>=65536) throw std::runtime_error("vocab too large for uint16 token cache");
}

std::vector<uint16_t> DataLoader::build_text(const std::string& path) const
{
    std::ifstream f(path);
    if(!f) throw std::runtime_error("cannot open "+path);

    std::vector<uint16_t> out;
    std::string line;
    size_t lines=0;
    while(std::getline(f,line))
    {
        line+='\n';
        for(int t:tok.encode(line)) out.push_back((uint16_t)t);
        if(++lines%200000==0) std::cout<<"[data] "<<path<<": "<<lines<<" lines, "<<out.size()<<" tokens\r"<<std::flush;
    }
    std::cout<<"[data] "<<path<<": "<<lines<<" lines, "<<out.size()<<" tokens\n";
    return out;
}

std::vector<uint16_t> DataLoader::build_dialogue(const std::string& path) const
{
    std::ifstream f(path);
    if(!f) throw std::runtime_error("cannot open "+path);

    int USER=tok.token_to_id("<|user|>"),BOT=tok.token_to_id("<|bot|>"),EOT=tok.token_to_id("<|endoftext|>");
    if(USER<0||BOT<0||EOT<0) throw std::runtime_error("tokenizer missing conversational special tokens");

    std::vector<uint16_t> out;
    std::string line;
    size_t dlg=0;
    while(std::getline(f,line))
    {
        std::vector<std::string> turns;
        size_t p=0;
        while(p<=line.size())
        {
            size_t q=line.find("__eou__",p);
            std::string t=(q==std::string::npos)?line.substr(p):line.substr(p,q-p);
            size_t a=t.find_first_not_of(" \t\r\n"),b=t.find_last_not_of(" \t\r\n");
            if(a!=std::string::npos) turns.push_back(t.substr(a,b-a+1));
            if(q==std::string::npos) break;
            p=q+7;
        }
        if(turns.empty()) continue;

        for(size_t i=0;i<turns.size();i++)
        {
            out.push_back((uint16_t)(i%2==0?USER:BOT));
            for(int t:tok.encode(" "+turns[i])) out.push_back((uint16_t)t);
        }
        out.push_back((uint16_t)EOT);
        if(++dlg%20000==0) std::cout<<"[data] "<<path<<": "<<dlg<<" dialogues, "<<out.size()<<" tokens\r"<<std::flush;
    }
    std::cout<<"[data] "<<path<<": "<<dlg<<" dialogues, "<<out.size()<<" tokens\n";
    return out;
}

std::vector<uint16_t> DataLoader::build_supervised(const std::string& path,std::vector<uint8_t>& sup) const
{
    std::ifstream f(path);
    if(!f) throw std::runtime_error("cannot open "+path);

    int USER=tok.token_to_id("<|user|>"),BOT=tok.token_to_id("<|bot|>"),EOT=tok.token_to_id("<|endoftext|>");
    if(USER<0||BOT<0||EOT<0) throw std::runtime_error("tokenizer missing conversational special tokens");

    std::vector<uint16_t> out;
    std::string line;
    size_t n=0;
    while(std::getline(f,line))
    {
        size_t tab=line.find('\t');
        if(tab==std::string::npos) continue;
        std::string prompt=line.substr(0,tab),completion=line.substr(tab+1);

        out.push_back((uint16_t)USER);sup.push_back(0);
        for(int t:tok.encode(" "+prompt)){out.push_back((uint16_t)t);sup.push_back(0);}

        out.push_back((uint16_t)BOT);sup.push_back(1);
        for(int t:tok.encode(" "+completion)){out.push_back((uint16_t)t);sup.push_back(1);}

        out.push_back((uint16_t)EOT);sup.push_back(0);

        if(++n%5000==0) std::cout<<"[data] "<<path<<": "<<n<<" examples, "<<out.size()<<" tokens\r"<<std::flush;
    }
    std::cout<<"[data] "<<path<<": "<<n<<" examples, "<<out.size()<<" tokens\n";
    return out;
}

std::vector<uint16_t> DataLoader::load_or_build(const std::string& txt,const std::string& cache,bool dialogue) const
{
    std::ifstream cf(cache,std::ios::binary);
    if(cf)
    {
        char m[4]={0,0,0,0};int32_t v=0;uint64_t n=0;
        cf.read(m,4);cf.read((char*)&v,4);cf.read((char*)&n,8);
        if(cf&&m[0]=='D'&&m[1]=='L'&&m[2]=='0'&&m[3]=='1'&&v==tok.size())
        {
            std::vector<uint16_t> out(n);
            cf.read((char*)out.data(),(std::streamsize)(n*2));
            if(cf){std::cout<<"[data] cache "<<cache<<" ("<<n<<" tokens)\n";return out;}
        }
        std::cout<<"[data] cache "<<cache<<" stale or invalid, rebuilding\n";
    }

    std::vector<uint16_t> out=dialogue?build_dialogue(txt):build_text(txt);

    std::ofstream of(cache,std::ios::binary);
    if(of)
    {
        const char m[4]={'D','L','0','1'};int32_t v=tok.size();uint64_t n=out.size();
        of.write(m,4);of.write((char*)&v,4);of.write((char*)&n,8);
        of.write((const char*)out.data(),(std::streamsize)(out.size()*2));
    }
    else std::cerr<<"[data] warning: cannot write cache "<<cache<<"\n";
    return out;
}

void DataLoader::add_source(std::vector<uint16_t>&& toks,double weight)
{
    if((long long)toks.size()<(long long)block+1) throw std::runtime_error("corpus has fewer than block+1 tokens");
    sources.push_back({std::move(toks),{},weight});
    weights.push_back(weight);
}

void DataLoader::add_text(const std::string& txt,const std::string& cache,double weight)
{
    add_source(load_or_build(txt,cache,false),weight);
}

void DataLoader::add_dialogue(const std::string& txt,const std::string& cache,double weight)
{
    add_source(load_or_build(txt,cache,true),weight);
}

void DataLoader::add_supervised(const std::string& txt,const std::string& cache,double weight)
{
    std::vector<uint16_t> toks;
    std::vector<uint8_t> sup;
    bool loaded=false;

    std::ifstream cf(cache,std::ios::binary);
    if(cf)
    {
        char m[4]={0,0,0,0};int32_t v=0;uint64_t n=0;
        cf.read(m,4);cf.read((char*)&v,4);cf.read((char*)&n,8);
        if(cf&&m[0]=='D'&&m[1]=='L'&&m[2]=='0'&&m[3]=='2'&&v==tok.size())
        {
            toks.resize(n);sup.resize(n);
            cf.read((char*)toks.data(),(std::streamsize)(n*2));
            cf.read((char*)sup.data(),(std::streamsize)n);
            if(cf){std::cout<<"[data] cache "<<cache<<" ("<<n<<" tokens)\n";loaded=true;}
        }
        if(!loaded) std::cout<<"[data] cache "<<cache<<" stale or invalid, rebuilding\n";
    }

    if(!loaded)
    {
        toks=build_supervised(txt,sup);
        std::ofstream of(cache,std::ios::binary);
        if(of)
        {
            const char m[4]={'D','L','0','2'};int32_t v=tok.size();uint64_t n=toks.size();
            of.write(m,4);of.write((char*)&v,4);of.write((char*)&n,8);
            of.write((const char*)toks.data(),(std::streamsize)(toks.size()*2));
            of.write((const char*)sup.data(),(std::streamsize)sup.size());
        }
        else std::cerr<<"[data] warning: cannot write cache "<<cache<<"\n";
    }

    if((long long)toks.size()<(long long)block+1) throw std::runtime_error("corpus has fewer than block+1 tokens");
    sources.push_back({std::move(toks),std::move(sup),weight});
    weights.push_back(weight);
}

void DataLoader::next(Tensor& X,Tensor& Y)
{
    if(sources.empty()) throw std::runtime_error("no data sources added");

    std::discrete_distribution<int> pick(weights.begin(),weights.end());
    for(int b=0;b<batch;b++)
    {
        const Source& s=sources[pick(rng)];
        std::uniform_int_distribution<size_t> off(0,s.toks.size()-block-1);
        size_t o=off(rng);
        for(int t=0;t<block;t++)
        {
            hX[b*block+t]=(float)s.toks[o+t];
            if(s.supervise.empty()||s.supervise[o+t]) hY[b*block+t]=(float)s.toks[o+t+1];
            else hY[b*block+t]=-100.0f;
        }
    }
    X.copy_from_host(hX.data());
    Y.copy_from_host(hY.data());
}

size_t DataLoader::total_tokens() const
{
    size_t n=0;
    for(auto& s:sources) n+=s.toks.size();
    return n;
}

long long DataLoader::steps_per_epoch() const
{
    return (long long)(total_tokens()/((size_t)batch*(size_t)block));
}

std::string DataLoader::rng_state() const
{
    std::ostringstream os;os<<rng;return os.str();
}

void DataLoader::set_rng_state(const std::string& s)
{
    std::istringstream is(s);is>>rng;
}
