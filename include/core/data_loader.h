#pragma once
#include "tensor.h"
#include "tokenizer.h"
#include <vector>
#include <string>
#include <random>
#include <cstdint>

class DataLoader
{
    private:
        BPETokenizer& tok;
        int block,batch;
        std::mt19937_64 rng;

        struct Source{std::vector<uint16_t> toks;std::vector<uint8_t> supervise;double weight;};
        std::vector<Source> sources;
        std::vector<double> weights;
        std::vector<float> hX,hY;

        std::vector<uint16_t> build_text(const std::string& path) const;
        std::vector<uint16_t> build_dialogue(const std::string& path) const;
        std::vector<uint16_t> build_supervised(const std::string& path,std::vector<uint8_t>& sup) const;
        std::vector<uint16_t> load_or_build(const std::string& txt,const std::string& cache,bool dialogue) const;
        void add_source(std::vector<uint16_t>&& toks,double weight);

    public:
        DataLoader(BPETokenizer& t,int block,int batch,uint64_t seed=1234);

        void add_text(const std::string& txt,const std::string& cache,double weight=1.0);
        void add_dialogue(const std::string& txt,const std::string& cache,double weight=1.0);
        void add_supervised(const std::string& txt,const std::string& cache,double weight=1.0);

        void next(Tensor& X,Tensor& Y);

        size_t total_tokens() const;
        long long steps_per_epoch() const;

        std::string rng_state() const;
        void set_rng_state(const std::string& s);
};
