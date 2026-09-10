#include "../../include/core/tokenizer.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace
{
    inline bool is_sp(unsigned char c){return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\v'||c=='\f';}
    inline bool is_dig(unsigned char c){return c>='0'&&c<='9';}
    inline bool is_let(unsigned char c){return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||c>=0x80;}
    inline bool is_oth(unsigned char c){return !is_sp(c)&&!is_dig(c)&&!is_let(c);}
    inline long long pk(int a,int b){return ((long long)a<<32)|(unsigned int)b;}
    const char* SPECIAL[]={"<|endoftext|>","<|user|>","<|bot|>","<|pad|>"};
    const int NUM_SPECIAL=4;
}

std::vector<std::string> BPETokenizer::pretokenize(const std::string& t) const
{
    std::vector<std::string> output;
    size_t n=t.size(),i=0;
    while(i<n)
    {
        unsigned char c=t[i];
        if(c=='\'' && i+1<n)
        {
            std::string s=(i+3<=n)?t.substr(i,3):"";
            if(s=="'re"||s=="'ve"||s=="'ll"){output.push_back(s);i+=3;continue;}
            unsigned char d=t[i+1];
            if(d=='s'||d=='t'||d=='m'||d=='d'){output.push_back(t.substr(i,2));i+=2;continue;}
        }

        bool lead=(c==' ' && i+1<n && !is_sp((unsigned char)t[i+1]));
        size_t start=i,k=lead?i+1:i;
        unsigned char c0=t[k];

        if(is_let(c0)){i=k;while(i<n && is_let((unsigned char)t[i]))i++;output.push_back(t.substr(start,i-start));continue;}
        if(is_dig(c0)){i=k;while(i<n && is_dig((unsigned char)t[i]))i++;output.push_back(t.substr(start,i-start));continue;}
        if(is_oth(c0)){i=k;while(i<n && is_oth((unsigned char)t[i]))i++;output.push_back(t.substr(start,i-start));continue;}

        size_t j=i;while(j<n && is_sp((unsigned char)t[j]))j++;
        if(j<n&&t[j-1]==' '&&j-1>i){output.push_back(t.substr(i,j-1-i));i=j-1;}
        else{output.push_back(t.substr(i,j-i));i=j;}
    }
    return output;
}

void BPETokenizer::finalize()
{
    vocab=256+(int)merge.size();
    total=vocab+NUM_SPECIAL;

    id.assign(vocab,{});
    for(int i=0;i<256;i++) id[i]={(uint8_t)i};
    for(size_t i=0;i<merge.size();i++)
    {
        std::vector<uint8_t> temp=id[merge[i].first];
        temp.insert(temp.end(),id[merge[i].second].begin(),id[merge[i].second].end());
        id[256+i]=std::move(temp);
    }

    rank.clear();
    for(size_t i=0;i<merge.size();i++) rank[pk(merge[i].first,merge[i].second)]=(int)i;

    id_token.clear();token_id.clear();
    for(int i=0;i<NUM_SPECIAL;i++)
    {
        id_token[vocab+i]=SPECIAL[i];
        token_id[SPECIAL[i]]=vocab+i;
    }
}

void BPETokenizer::train(const std::vector<std::string>& path,int size,size_t max)
{
    int K=size-256-NUM_SPECIAL;
    if(K<0) throw std::invalid_argument("target vocab too small");

    std::unordered_map<std::string,long long> wf;
    size_t seen=0;
    for(auto& p:path)
    {
        std::ifstream f(p);
        if(!f) throw std::runtime_error("can not open "+p);
        std::string line;
        while(seen<max && std::getline(f,line)) for(auto& w:pretokenize(line)) {wf[w]++;if(++seen>=max)break;}
    }
    std::cout <<"[bpe] "<<wf.size()<<" unique words from "<<seen<<" tokens"<<std::endl;

    std::vector<std::vector<int>> ws;
    std::vector<long long> wc;
    ws.reserve(wf.size());wc.reserve(wf.size());
    for(auto& kv:wf)
    {
        std::vector<int> s;
        s.reserve(kv.first.size());
        for(unsigned char c:kv.first) s.push_back((int)c);
        ws.push_back(std::move(s));
        wc.push_back(kv.second);
    }

    std::unordered_map<long long,long long> pc;
    for(size_t i=0;i<ws.size();i++) for(size_t j=0;j+1<ws[i].size();j++) pc[pk(ws[i][j],ws[i][j+1])]+=wc[i];
    merge.clear();
    merge.reserve(K);

    for(int m=0;m<K;m++)
    {
        long long best=0,bestkey=-1;
        for(auto& kv:pc) if(kv.second>best || (kv.second==best && bestkey>=0 && kv.first<bestkey)) {best=kv.second;bestkey=kv.first;}

        if(bestkey<0 || best<=0) break;

        int a=(int)(bestkey>>32),b=(int)(unsigned int)bestkey,nid=256+m;
        merge.push_back({a,b});

        for(size_t w=0;w<ws.size();w++)
        {
            auto& s=ws[w];
            if(s.size()<2) continue;

            std::vector<int> ns;
            ns.reserve(s.size());
            for(size_t k=0;k<s.size();)
            {
                if(k+1<s.size() && s[k]==a && s[k+1]==b){ns.push_back(nid);k+=2;}
                else {ns.push_back(s[k]);k++;}
            }
            if(ns.size()==s.size()) continue;

            long long fr=wc[w];
            for(size_t k=0;k+1<s.size();k++) pc[pk(s[k],s[k+1])]-=fr;
            for(size_t k=0;k+1<ns.size();k++) pc[pk(ns[k],ns[k+1])]+=fr;
            s=std::move(ns);
        }
        pc.erase(bestkey);
        if((m+1)%1000==0) std::cout<<"[bpe] "<<m+1<<"/"<<K<<" merges"<<std::endl;
    }
    finalize();
}

std::vector<int> BPETokenizer::bpe(const std::string& piece) const
{
    auto ci=cache.find(piece);
    if(ci!=cache.end()) return ci->second;

    std::vector<int> s;
    s.reserve(piece.size());
    for(unsigned char c:piece) s.push_back((int)c);

    while(s.size()>=2)
    {
        int best_rank=-1;long long best_key=0;
        for(size_t i=0;i+1<s.size();i++)
        {
            auto it=rank.find(pk(s[i],s[i+1]));
            if(it!=rank.end() && (best_rank<0 || it->second<best_rank)){best_rank=it->second;best_key=pk(s[i],s[i+1]);}
        }
        if(best_rank<0) break;

        int a=(int)(best_key>>32),b=(int)(unsigned int)best_key,nid=256+best_rank;
        std::vector<int> ns;
        ns.reserve(s.size());
        for(size_t i=0;i<s.size();)
        {
            if(i+1<s.size() && s[i]==a && s[i+1]==b){ns.push_back(nid);i+=2;}
            else{ns.push_back(s[i]);i++;}
        }
        s=std::move(ns);
    }

    cache[piece]=s;
    return s;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const
{
    std::vector<int> out;
    size_t i=0,n=text.size();
    while(i<n)
    {
        size_t best_pos=std::string::npos;int best_id=-1;size_t best_len=0;
        for(auto& kv:token_id)
        {
            size_t p=text.find(kv.first,i);
            if(p!=std::string::npos && (p<best_pos || (p==best_pos && kv.first.size()>best_len))){best_pos=p;best_id=kv.second;best_len=kv.first.size();}
        }

        size_t end=(best_pos==std::string::npos)?n:best_pos;
        if(end>i)
            for(auto& piece:pretokenize(text.substr(i,end-i)))
            {
                auto ids=bpe(piece);
                out.insert(out.end(),ids.begin(),ids.end());
            }

        if(best_pos==std::string::npos) break;
        out.push_back(best_id);
        i=best_pos+best_len;
    }
    return out;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const
{
    std::string out;
    for(int x:ids)
    {
        if(x>=0 && x<vocab) for(uint8_t b:id[x]) out+=(char)b;
        else
        {
            auto it=id_token.find(x);
            if(it!=id_token.end()) out+=it->second;
        }
    }
    return out;
}

int BPETokenizer::token_to_id(const std::string& token) const
{
    auto it=token_id.find(token);
    return it!=token_id.end()?it->second:-1;
}

void BPETokenizer::save(const std::string& path) const
{
    std::ofstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path+" for writing");
    const char magic[4]={'B','P','E','1'};
    f.write(magic,4);
    int32_t nm=(int32_t)merge.size();
    f.write((char*)&nm,4);
    for(auto& e:merge){int32_t a=e.first,b=e.second;f.write((char*)&a,4);f.write((char*)&b,4);}
}

void BPETokenizer::load(const std::string& path)
{
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path+" for reading");
    char magic[4];f.read(magic,4);
    if(magic[0]!='B'||magic[1]!='P'||magic[2]!='E'||magic[3]!='1') throw std::runtime_error("bad tokenizer file: "+path);
    int32_t nm=0;f.read((char*)&nm,4);
    merge.clear();merge.reserve(nm);
    for(int i=0;i<nm;i++){int32_t a=0,b=0;f.read((char*)&a,4);f.read((char*)&b,4);merge.push_back({a,b});}
    finalize();
}