#include "../include/core/gpt.h"
#include "../include/core/optimizer.h"
#include "../include/core/loss.h"
#include "../include/core/tokenizer.h"
#include "../include/core/data_loader.h"
#include "../include/core/checkpoint.h"
#include "../include/core/generator.h"
#include "../include/core/memory.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

struct Config
{
    int dmodel=384,heads=6,dff=1536,layers=6,max_seq=256,block=128;
    float dropout=0.0f;

    float lr=4e-4f,lr_min=4e-5f,wd=0.01f,beta2=0.95f;
    int batch=48;
    long long max_steps=20000,warmup=500;

    int log_every=20,ckpt_every=500,sample_every=500;

    bool bf16=false;

    std::string tok_path="data/ir_tok.bin";
    std::string ckpt_path="data/ir_ckpt.bin";
    std::string weights_path="data/ir_model.bin";
    std::string text="data/ir_pairs.txt";
    std::string text_cache="data/ir_pairs.cache";
};

static void apply_args(Config& c,int argc,char** argv)
{
    for(int i=1;i<argc;i++)
    {
        std::string a=argv[i];
        size_t eq=a.find('=');
        if(eq==std::string::npos) continue;
        std::string k=a.substr(0,eq),v=a.substr(eq+1);
        if(k=="batch") c.batch=std::stoi(v);
        else if(k=="block") c.block=std::stoi(v);
        else if(k=="steps") c.max_steps=std::stoll(v);
        else if(k=="warmup") c.warmup=std::stoll(v);
        else if(k=="dmodel") c.dmodel=std::stoi(v);
        else if(k=="heads") c.heads=std::stoi(v);
        else if(k=="dff") c.dff=std::stoi(v);
        else if(k=="layers") c.layers=std::stoi(v);
        else if(k=="max_seq") c.max_seq=std::stoi(v);
        else if(k=="lr") c.lr=std::stof(v);
        else if(k=="wd") c.wd=std::stof(v);
        else if(k=="dropout") c.dropout=std::stof(v);
        else if(k=="ckpt") c.ckpt_path=v;
        else if(k=="tok") c.tok_path=v;
        else if(k=="weights") c.weights_path=v;
        else if(k=="text") c.text=v;
        else if(k=="text_cache") c.text_cache=v;
        else if(k=="ckpt_every") c.ckpt_every=std::stoi(v);
        else if(k=="sample_every") c.sample_every=std::stoi(v);
        else if(k=="log_every") c.log_every=std::stoi(v);
        else if(k=="bf16") c.bf16=(v=="1"||v=="true");
        else std::cerr<<"[train] unknown arg: "<<k<<"\n";
    }
}

static float schedule_lr(const Config& c,long long step)
{
    if(step<c.warmup) return c.lr*(float)(step+1)/(float)c.warmup;
    float prog=(float)(step-c.warmup)/(float)std::max<long long>(1,c.max_steps-c.warmup);
    prog=std::min(1.0f,prog);
    return c.lr_min+0.5f*(c.lr-c.lr_min)*(1.0f+std::cos(3.14159265f*prog));
}

static void sample_ir(GPT& model,BPETokenizer& tok,int max_seq,long long step)
{
    static const std::vector<std::string> questions={
        "What is the largest SMU based on VolumeInvoiced?",
        "Who is the Area Sales Manager for customer code ABC123",
        "set Cancelled to 1 for OBD91541,OBD82743"
    };

    int USER=tok.token_to_id("<|user|>"),BOT=tok.token_to_id("<|bot|>"),EOT=tok.token_to_id("<|endoftext|>");

    model.eval();
    for(auto& q:questions)
    {
        std::vector<int> prompt;
        prompt.push_back(USER);
        for(int t:tok.encode(" "+q)) prompt.push_back(t);
        prompt.push_back(BOT);

        GenConfig gc;gc.max_new_tokens=80;gc.seed=(uint64_t)step;
        std::vector<int> out=generate(model,prompt,max_seq,gc,EOT);
        std::cout<<"[sample] q: "<<q<<"\n[sample] ir: "<<tok.decode(out)<<"\n";
    }
    model.train();
}

int main(int argc,char** argv)
{
    Config cfg;
    apply_args(cfg,argc,argv);

    BPETokenizer tok;
    tok.load(cfg.tok_path);
    int V=tok.size();
    std::cout<<"[train] vocab="<<V<<" | model "<<cfg.layers<<"L d"<<cfg.dmodel<<" h"<<cfg.heads<<" ff"<<cfg.dff
             <<" | block="<<cfg.block<<" batch="<<cfg.batch<<"\n";

    DataLoader dl(tok,cfg.block,cfg.batch,1234);
    dl.add_supervised(cfg.text,cfg.text_cache,1.0);
    std::cout<<"[train] total_tokens="<<dl.total_tokens()<<" steps/epoch="<<dl.steps_per_epoch()<<"\n";

    GPT model(V,cfg.dmodel,cfg.heads,cfg.dff,cfg.layers,cfg.max_seq,cfg.dropout,cfg.bf16?DType::BF16:DType::F32);
    Adam opt(model.get_layers(),cfg.lr,0.9f,cfg.beta2,1e-8f,cfg.wd);
    SparseCrossEntropyLoss loss_fn;
    model.compile(&opt,&loss_fn);

    long long step=0;
    float best=1e9f;
    {
        std::ifstream f(cfg.ckpt_path,std::ios::binary);
        if(f.good())
        {
            f.close();
            load_checkpoint(cfg.ckpt_path,model,opt,step,best,dl);
            std::cout<<"[train] resumed from step "<<step<<" (best loss "<<best<<")\n";
        }
    }

    model.train();
    Tensor X=Tensor::zeros({cfg.batch,cfg.block}),Y=Tensor::zeros({cfg.batch,cfg.block});

    float run=0.0f;
    int rc=0;
    for(;step<cfg.max_steps;step++)
    {
        float lr=schedule_lr(cfg,step);
        opt.set_lr(lr);

        dl.next(X,Y);
        float l=model.train_step(X,Y);
        run+=l;rc++;

        if((step+1)%cfg.log_every==0)
        {
            std::cout<<"step "<<step+1<<"/"<<cfg.max_steps<<" | loss "<<run/rc<<" | lr "<<lr<<std::endl;
            run=0.0f;rc=0;
        }

        if((step+1)%cfg.ckpt_every==0)
        {
            if(l<best) best=l;
            save_checkpoint(cfg.ckpt_path,model,opt,step+1,best,dl);
            save_weights_only(cfg.weights_path,model);
            std::cout<<"[train] checkpoint @ step "<<step+1<<"\n";
        }

        if((step+1)%cfg.sample_every==0) sample_ir(model,tok,cfg.max_seq,step);
    }

    save_checkpoint(cfg.ckpt_path,model,opt,step,best,dl);
    save_weights_only(cfg.weights_path,model);
    std::cout<<"[train] done at step "<<step<<"\n";
    clear_memory_pool();
    return 0;
}