#pragma once
// GPU-assisted sampling (mirrors llama.cpp's approach of doing the heavy
// softmax / top-k work on the GPU).
//
// The kernel computes the softmax distribution over the full vocab and keeps,
// per thread, its local top-16 probabilities. With 512 threads this yields
// SAM_CAND_PER_SEQ = 8192 candidates per sequence - a superset that always
// contains the true top-16 and, practically, the entire top-p mass. Only these
// candidates are copied back to the host (32 KB instead of 600 KB of raw
// logits); the host then applies temperature scaling, top-k, top-p and the
// final RNG draw over at most 8192 items (sub-millisecond).

#include "common.cuh"

constexpr int SAM_THREADS = 512;
constexpr int SAM_TOP_PER_THREAD = 16;
constexpr int SAM_CAND_PER_SEQ = SAM_THREADS * SAM_TOP_PER_THREAD;  // 8192

// Greedy sampling entirely on the GPU. Only M token ids are copied to the
// host, instead of M*n_vocab FP32 logits.
void sample_argmax(const float * logits, int M, int n_vocab,
                   int * token_ids, cudaStream_t stream = 0);

// For each of the M sequences, fill cand_vals[m*CAND + i] = softmax prob and
// cand_idx[m*CAND + i] = vocab index of the i-th candidate (top-16 per thread).
// Runs on the given stream; callers must ensure `logits` is ready on it.
void sample_topk_probs(const float * logits, int M, int n_vocab,
                       float * cand_vals, int * cand_idx, cudaStream_t stream = 0);
