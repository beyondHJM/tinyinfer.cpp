#pragma once
// Qwen3 model structure and weight loading.
// Reads hyperparameters from the GGUF (no hardcoding), asserts BF16 weights.

#include "cuda/common.cuh"
#include "gguf_reader.h"

#include <string>
#include <vector>

struct Qwen3Config {
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int n_ff = 0;
    int n_vocab = 0;
    float rope_theta = 0.0f;
    float norm_eps = 0.0f;
};

struct Qwen3Layer {
    // norms (FP32)
    float * attn_norm = nullptr;   // [n_embd]
    float * ffn_norm  = nullptr;   // [n_embd]
    float * q_norm    = nullptr;   // [head_dim]
    float * k_norm    = nullptr;   // [head_dim]
    // weights as W^T, row-major [N, K], BF16
    bf16 * wq  = nullptr;          // [n_head*head_dim, n_embd]
    bf16 * wk  = nullptr;          // [n_head_kv*head_dim, n_embd]
    bf16 * wv  = nullptr;          // [n_head_kv*head_dim, n_embd]
    bf16 * wo  = nullptr;          // [n_embd, n_head*head_dim]
    bf16 * w_gate = nullptr;       // [n_ff, n_embd]
    bf16 * w_up   = nullptr;       // [n_ff, n_embd]
    bf16 * w_down = nullptr;       // [n_embd, n_ff]
};

struct Qwen3Model {
    Qwen3Config cfg;

    bf16  * tok_embd = nullptr;    // [n_vocab, n_embd] (also the LM head)
    float * output_norm = nullptr; // [n_embd]
    std::vector<Qwen3Layer> layers;

    bool load(const std::string & gguf_path);
    void free_all();
    ~Qwen3Model() { free_all(); }

    // Load one tensor from the GGUF into host memory (returns host buffer).
    static bool load_host_tensor(const GgufReader & r, const std::string & name,
                                 std::vector<uint8_t> & out, size_t & nbytes);
};
