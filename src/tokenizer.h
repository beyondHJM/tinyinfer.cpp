#pragma once
// Qwen3 (Qwen2-style GPT2 BPE) tokenizer.
// Cropped/adapted from llama.cpp src/llama-vocab.cpp (MIT) and
// src/unicode.cpp (MIT). Self-contained; only depends on the standard library.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Tokenizer {
public:
    // Token types (GGUF tokenizer.ggml.token_type == LLAMA_TOKEN_TYPE_*).
    enum : uint32_t {
        TOKEN_UNDEFINED    = 0,
        TOKEN_NORMAL       = 1,
        TOKEN_UNKNOWN      = 2,
        TOKEN_CONTROL      = 3,
        TOKEN_USER_DEFINED = 4,
        TOKEN_UNUSED       = 5,
        TOKEN_BYTE         = 6,
    };

    bool load_tokens(const std::vector<std::string> & tokens,
                     const std::vector<uint32_t> & types);
    bool load_merges(const std::vector<std::string> & merges);

    void set_eos_id(int32_t id) { eos_id_ = id; }
    int32_t eos_id() const { return eos_id_; }

    size_t n_tokens() const { return tokens_.size(); }

    // add_special: prepend BOS if the vocab requests it (Qwen3 has none).
    // parse_special: recognize <|...|> control tokens embedded in text.
    std::vector<int32_t> tokenize(const std::string & text,
                                  bool add_special = false,
                                  bool parse_special = false) const;

    // Decode one token id to its UTF-8 piece (for streaming output).
    std::string token_to_piece(int32_t id, bool special = true) const;

    bool is_eog(int32_t id) const { return id == eos_id_; }
    bool is_special(int32_t id) const {
        if (id < 0 || (size_t) id >= types_.size()) return false;
        uint32_t t = types_[id];
        return t == TOKEN_UNKNOWN || t == TOKEN_CONTROL;
    }

private:
    struct Symbol {
        int prev = -1;
        int next = -1;
        const char * text = nullptr;
        size_t n = 0;
    };

    struct Bigram {
        int left = 0;
        int right = 0;
        std::string text;
        int rank = 0;
        size_t size = 0;
    };

    std::vector<std::string> tokens_;
    std::vector<uint32_t> types_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;   // key: first + " " + second
    std::vector<std::string> byte_decoder_cpts_;       // cpt string -> byte
    std::unordered_map<uint8_t, std::string> byte_to_utf8_;  // GPT-2 byte encoder
    std::unordered_map<std::string, uint8_t> utf8_to_byte_;
    int32_t eos_id_ = -1;
};
