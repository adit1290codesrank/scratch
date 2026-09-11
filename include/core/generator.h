#pragma once
#include <string>
#include <vector>
#include <cstdint>

class GPT;
class BPETokenizer;

struct GenConfig
{
    int max_new_tokens=200;
    int top_k=40;
    float temperature=0.9f;
    uint64_t seed=1234;
};

std::vector<int> generate(GPT& model,const std::vector<int>& prompt_ids,int max_seq,const GenConfig& cfg,int eot_id,int stop_id1=-1,int stop_id2=-1);
std::string chat_generate(GPT& model,BPETokenizer& tok,const std::vector<std::string>& history,int max_seq,const GenConfig& cfg);
