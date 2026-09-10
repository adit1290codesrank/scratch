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
    int dmodel=512,heads=8,dff=2048,layers=6,max_seq=1024,block=512;
    float dropout=0.1f;

    float lr=3e-4f,lr_min=3e-5f,wd=0.1f,beta2=0.95f;
    int batch=24;
    long long max_steps=200000,warmup=2000;

    int log_every=20,ckpt_every=1000,sample_every=1000;

    std::string tok_path="data/tok.bin";
    std::string ckpt_path="data/ckpt.bin";
    std::string weights_path="data/model.bin";
    std::string text="data/TinyStories-train.txt";
    std::string text_cache="data/ts.tokens";
    std::string dialogue="data/dialogues_text.txt";
    std::string dialogue_cache="data/dlg.tokens";
    double dialogue_weight=0.30;
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
        else if(k=="dlgw") c.dialogue_weight=std::stod(v);
        else if(k=="ckpt") c.ckpt_path=v;
        else if(k=="tok") c.tok_path=v;
        else if(k=="weights") c.weights_path=v;
        else if(k=="text") c.text=v;
        else if(k=="text_cache") c.text_cache=v;
        else if(k=="dialogue") c.dialogue=v;
        else if(k=="dialogue_cache") c.dialogue_cache=v;
        else if(k=="ckpt_every") c.ckpt_every=std::stoi(v);
        else if(k=="sample_every") c.sample_every=std::stoi(v);
        else if(k=="log_every") c.log_every=std::stoi(v);
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
    dl.add_text(cfg.text,cfg.text_cache,1.0-cfg.dialogue_weight);
    {
        std::ifstream d(cfg.dialogue);
        if(d) dl.add_dialogue(cfg.dialogue,cfg.dialogue_cache,cfg.dialogue_weight);
        else std::cout<<"[train] no dialogue file at "<<cfg.dialogue<<", training on text only\n";
    }
    std::cout<<"[train] total_tokens="<<dl.total_tokens()<<" steps/epoch="<<dl.steps_per_epoch()<<"\n";

    GPT model(V,cfg.dmodel,cfg.heads,cfg.dff,cfg.layers,cfg.max_seq,cfg.dropout);
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

        if((step+1)%cfg.sample_every==0)
        {
            model.eval();
            GenConfig gc;gc.max_new_tokens=60;gc.seed=(uint64_t)step;
            std::string r=chat_generate(model,tok,{"hi, how are you?"},cfg.max_seq,gc);
            std::cout<<"[sample] user: hi, how are you?\n[sample] bot: "<<r<<"\n";
            model.train();
        }
    }

    save_checkpoint(cfg.ckpt_path,model,opt,step,best,dl);
    save_weights_only(cfg.weights_path,model);
    std::cout<<"[train] done at step "<<step<<"\n";
    clear_memory_pool();
    return 0;
}
