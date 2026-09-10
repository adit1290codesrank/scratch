#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

class BPETokenizer
{
    private:
        std::unordered_map<long long,int> rank;
        std::vector<std::pair<int,int>> merge;
        std::vector<std::vector<uint8_t>> id;
        std::unordered_map<int,std::string> id_token;
        std::unordered_map<std::string,int> token_id;
        mutable std::unordered_map<std::string,std::vector<int>> cache;
        int vocab=0,total=0;

        std::vector<std::string> pretokenize(const std::string& text) const;
        std::vector<int> bpe(const std::string& piece) const;
        void finalize();

    public:
        BPETokenizer(){};

        void train(const std::vector<std::string>& path,int size,size_t max=20000000);
        void save(const std::string& path) const;
        void load(const std::string& path);

        std::vector<int> encode(const std::string& text) const;
        std::string decode(const std::vector<int>& id) const;

        int token_to_id(const std::string& token) const;
        int size() const{return total;}

};
