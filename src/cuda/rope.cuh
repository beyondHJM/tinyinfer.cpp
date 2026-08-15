#pragma once

#include "common.cuh"

// GPT-NeoX style RoPE (adjacent pairs) - same as llama.cpp for QWEN3.
// x/out: [rows, n_heads*head_dim]; positions: [rows].
void rope_neox(const float * x, const int * positions, int rows, int n_heads, int head_dim,
               float theta, float * out, cudaStream_t stream = 0);
