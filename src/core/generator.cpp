#include "../../include/core/generator.h"
#include "../../include/core/gpt.h"
#include "../../include/core/tensor.h"
#include "../../include/core/tokenizer.h"
#include <random>
#include <algorithm>
#include <cmath>

std::vector<int> generate(GPT& model,const std::vector<int>& prompt_ids,int max_seq,const GenConfig& cfg,int eot_id,int stop_id)
{
    model.eval();
    std::mt19937_64 rng(cfg.seed);

    std::vector<int> ctx=prompt_ids;
    std::vector<int> out;
    std::vector<float> row;

    for(int s=0;s<cfg.max_new_tokens;s++)
    {
        int T=(int)ctx.size();
        int start=T>max_seq?T-max_seq:0;
        int len=T-start;

        Tensor X=Tensor::zeros({1,len});
        std::vector<float> hx(len);
        for(int i=0;i<len;i++) hx[i]=(float)ctx[start+i];
        X.copy_from_host(hx.data());

        Tensor lg=model.logits(X);                        // [1,len,V]
        int V=lg.shape.back();
        Tensor last=lg.reshape({len,V}).slice(len-1,len); // [1,V] on device
        row.resize(V);
        last.copy_to_host(row.data());

        bool greedy=(cfg.temperature<=1e-6f||cfg.top_k==1);
        if(!greedy)
        {
            float inv=1.0f/cfg.temperature;
            for(float& x:row) x*=inv;
        }

        int k=(cfg.top_k>0&&cfg.top_k<V)?cfg.top_k:V;
        std::vector<int> idx(V);
        for(int i=0;i<V;i++) idx[i]=i;
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){return row[a]>row[b];});

        int next;
        if(greedy) next=idx[0];
        else
        {
            float mx=row[idx[0]];
            double sum=0;
            std::vector<double> p(k);
            for(int i=0;i<k;i++){p[i]=std::exp((double)(row[idx[i]]-mx));sum+=p[i];}
            std::uniform_real_distribution<double> U(0.0,sum);
            double r=U(rng),acc=0;
            next=idx[k-1];
            for(int i=0;i<k;i++){acc+=p[i];if(r<=acc){next=idx[i];break;}}
        }

        out.push_back(next);
        ctx.push_back(next);
        if(next==eot_id||next==stop_id) break;
    }
    return out;
}

std::string chat_generate(GPT& model,BPETokenizer& tok,const std::vector<std::string>& history,int max_seq,const GenConfig& cfg)
{
    int USER=tok.token_to_id("<|user|>");
    int BOT=tok.token_to_id("<|bot|>");
    int EOT=tok.token_to_id("<|endoftext|>");
    if(USER<0||BOT<0||EOT<0) return "[tokenizer missing chat special tokens]";

    std::vector<int> ids;
    for(size_t i=0;i<history.size();i++)
    {
        ids.push_back(i%2==0?USER:BOT);
        for(int t:tok.encode(" "+history[i])) ids.push_back(t);
    }
    ids.push_back(BOT);

    std::vector<int> gen=generate(model,ids,max_seq,cfg,EOT,USER);
    if(!gen.empty()&&(gen.back()==EOT||gen.back()==USER)) gen.pop_back();

    std::string s=tok.decode(gen);
    size_t a=s.find_first_not_of(" \t\r\n");
    return a==std::string::npos?"":s.substr(a);
}
