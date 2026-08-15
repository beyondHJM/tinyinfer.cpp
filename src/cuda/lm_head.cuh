#pragma once

#include "common.cuh"

// logits[M, vocab] = x[M, n_embd] @ w[vocab, n_embd]^T (BF16 weights).
void lm_head_forward(const float * x, int M, const bf16 * w, int vocab, int n_embd,
                     bf16 * xb, float * logits, cudaStream_t stream = 0);
