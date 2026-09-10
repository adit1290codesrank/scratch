#pragma once
#include <string>

class GPT;
class Optimizer;
class DataLoader;

struct GPTDims{int vocab,dmodel,heads,dff,layers,max_seq;};

// read just the architecture header of a weights-only file (see save_weights_only)
GPTDims read_gpt_dims(const std::string& path);

// full resumable training state: meta + loader RNG + model weights + optimizer moments
void save_checkpoint(const std::string& path,GPT& model,Optimizer& opt,long long step,float best_loss,const DataLoader& loader);
void load_checkpoint(const std::string& path,GPT& model,Optimizer& opt,long long& step,float& best_loss,DataLoader& loader);

// slim inference artifact: model weights only
void save_weights_only(const std::string& path,GPT& model);
void load_weights_only(const std::string& path,GPT& model);
