#pragma once

#include "common.cuh"

// Stage 1: per-token Q/K RMSNorm (per-head) + RoPE (NeoX) + write K/V into the
// KV cache. q/k/v: [M, nq]/[M, nk] FP32; q_out: [M, nq] FP32 (normalized+roped Q).
// kv_cache (per layer): [n_seqs, 2*n_kv*head_dim*max_seq] BF16; kv_stride per seq.
void qk_norm_store(const float * q, const float * k, const float * v,
                   int q_stride, int k_stride, int v_stride,
                   const float * q_norm, const float * k_norm,
                   float norm_eps, float rope_theta,
                   int head_dim, int n_head, int n_kv,
                   const int * positions, const int * seq_ids, int M,
                   bf16 * kv_cache, int kv_stride, int max_seq,
                   float * q_out, cudaStream_t stream = 0);

// Stage 2: GQA causal attention over the KV cache (all K/V already written).
// q: [M, nq] FP32 (normalized+roped); out: [M, nq] FP32.
void attention_compute(const float * q, const int * positions, const int * seq_ids, int M,
                       const bf16 * kv_cache, int kv_stride, int max_seq,
                       int head_dim, int n_head, int n_kv,
                       float * out, cudaStream_t stream = 0);

// Prefill variant: many tokens at once (M > n_seqs). One 16-thread group per
// token, each scanning its own full causal prefix with an online softmax and
// vectorized K/V reads - no decode-style sequence splitting, so prefill keeps
// full parallelism and skips the split/merge overhead.
void prefill_attention_compute(const float * q, const int * positions, const int * seq_ids,
                               int M, const bf16 * kv_cache, int kv_stride, int max_seq,
                               int head_dim, int n_head, int n_kv,
                               float * out, cudaStream_t stream = 0);
