#pragma once
// Batched prefill / decode inference loop with a BF16 KV cache.

#include "model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

class Inference {
public:
    ~Inference();

    // max_seq: max tokens per sequence (KV cache size), max_batch: n sequences.
    bool init(const Qwen3Model & model, int max_seq, int max_batch);
    void reset_kv();

    // Enable/disable CUDA Graph replay for the decode phase (off by default).
    // When enabled, each distinct decode batch size M is captured once (on the
    // first decode forward with that M) and replayed via cudaGraphLaunch.
    void set_use_cuda_graph(bool b) { use_cuda_graph_ = b; }
    bool use_cuda_graph() const { return use_cuda_graph_; }

    // One forward pass over n_tokens tokens. Each token belongs to seq_ids[i]
    // at position positions[i]. Copies the last-token logits of every sequence
    // to last_logits (host, [n_seqs * n_vocab]).
    //
    // If cand_probs/cand_idx are non-null, the full logits are NOT copied back;
    // instead the GPU softmax + top-k pass fills them with SAM_CAND_PER_SEQ
    // (prob, index) candidates per sequence (see sample.cuh), which is much
    // cheaper for host-side sampling.
    bool forward(const std::vector<int32_t> & tokens,
                 const std::vector<int> & positions,
                 const std::vector<int> & seq_ids,
                 int n_seqs,
                 std::vector<float> & last_logits,
                 std::vector<float> * cand_probs = nullptr,
                 std::vector<int> * cand_idx = nullptr);

    int n_vocab() const { return model_ ? model_->cfg.n_vocab : 0; }

private:
    void free_all();

    // Launch the full decoder kernel sequence (embed -> 28 layers -> output
    // norm -> gather -> lm head) on the given stream. No host/device copies and
    // no error checks: callers wrap it with the appropriate sync / checks.
    void launch_forward_kernels(int M, int n_seqs, cudaStream_t stream);

    // Return a CUDA Graph executable for a decode forward with M tokens,
    // capturing one on first use. Returns nullptr if graphs are unavailable
    // (e.g. too many distinct M values) - callers then fall back to eager.
    cudaGraphExec_t get_or_capture_decode_graph(int M, int n_seqs);

    // Debug helper: with QWEN_GRAPH_VERIFY=1, run one eager forward and one
    // graph replay on identical KV state and compare logits numerically.
    void verify_graph_vs_eager(int M, int n_seqs, cudaGraphExec_t exec);

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

    // GPU sampling candidates (see sample.cuh).
    float * cand_vals_ = nullptr;  // [max_batch, SAM_CAND_PER_SEQ]
    int   * cand_idx_ = nullptr;   // [max_batch, SAM_CAND_PER_SEQ]

    bf16 * kv_ = nullptr;        // [max_batch, n_layer, 2*n_kv*head_dim*max_seq]
    size_t kv_seq_stride_ = 0;
    size_t kv_layer_stride_ = 0;

    bool use_cuda_graph_ = false;
    cudaStream_t graph_stream_ = nullptr;             // stream used for capture + replay
    std::vector<std::pair<int, cudaGraphExec_t>> decode_graphs_; // M -> graph exec
};
