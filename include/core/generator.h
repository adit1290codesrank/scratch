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

// continue from a raw token sequence; stops on eot_id, stop_id, or length.
// returns only the newly generated tokens (terminator included if hit).
std::vector<int> generate(GPT& model,const std::vector<int>& prompt_ids,int max_seq,const GenConfig& cfg,int eot_id,int stop_id=-1);

// history = alternating user/bot turns (index 0 = first user turn).
// builds <|user|> .. <|bot|> .. <|user|> .. <|bot|> framing and completes the bot turn.
std::string chat_generate(GPT& model,BPETokenizer& tok,const std::vector<std::string>& history,int max_seq,const GenConfig& cfg);
