#pragma once
// Batched prefill / decode inference loop with a BF16 KV cache.

#include "model.h"

#include <cstdint>
#include <vector>

class Inference {
public:
    ~Inference();

    // max_seq: max tokens per sequence (KV cache size), max_batch: n sequences.
    bool init(const Qwen3Model & model, int max_seq, int max_batch);
    void reset_kv();

    // One forward pass over n_tokens tokens. Each token belongs to seq_ids[i]
    // at position positions[i]. Copies the last-token logits of every sequence
    // to last_logits (host, [n_seqs * n_vocab]).
    bool forward(const std::vector<int32_t> & tokens,
                 const std::vector<int> & positions,
                 const std::vector<int> & seq_ids,
                 int n_seqs,
                 std::vector<float> & last_logits);

    int n_vocab() const { return model_ ? model_->cfg.n_vocab : 0; }

private:
    void free_all();

    const Qwen3Model * model_ = nullptr;
    int max_seq_ = 0;
    int max_batch_ = 0;
    int max_tokens_ = 0;

    float * x_ = nullptr;        // [max_tokens, n_embd]
    float * xn_ = nullptr;       // [max_tokens, n_embd]
    bf16  * xb_ = nullptr;       // [max_tokens, n_embd]
    bf16  * attn_out_b_ = nullptr; // [max_tokens, nq]
    bf16  * ffn_b_ = nullptr;    // [max_tokens, n_ff]
    float * qkv_ = nullptr;      // [max_tokens, nq + 2*nk] fused Q,K,V
    float * qn_buf_ = nullptr;   // [max_tokens, nq] normalized+roped Q
    float * attn_out_ = nullptr; // [max_tokens, nq]
    float * o_ = nullptr;        // [max_tokens, n_embd]
    float * gu_ = nullptr;       // [max_tokens, 2*n_ff]
    float * y_ = nullptr;        // [max_tokens, n_ff] SwiGLU output
    bf16  * wgu_ = nullptr;      // [n_layer, 2*n_ff, n_embd] combined gate+up per layer
    bf16  * wqkv_ = nullptr;     // [n_layer, nq+2*nk, n_embd] fused Q,K,V per layer
    float * x_last_ = nullptr;   // [max_batch, n_embd]
    bf16  * x_last_b_ = nullptr; // [max_batch, n_embd]
    float * logits_ = nullptr;   // [max_batch, n_vocab]

    int * tok_dev_ = nullptr;
    int * pos_dev_ = nullptr;
    int * seq_dev_ = nullptr;
    int * last_idx_dev_ = nullptr;

    bf16 * kv_ = nullptr;        // [max_batch, n_layer, 2*n_kv*head_dim*max_seq]
    size_t kv_seq_stride_ = 0;
    size_t kv_layer_stride_ = 0;
};
