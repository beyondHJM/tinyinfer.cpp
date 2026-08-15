#pragma once

#include "common.cuh"

// BF16 tensor-core GEMM: C[M,N] = A[M,K] * B[N,K]^T
// A: row-major [M,K]; B (weights): row-major [N,K]; C: row-major [M,N] FP32.
// Requires M % 64 == 0 || M < 64 handled; N % 64 == 0 handled generically.
void gemm_bf16(const bf16 * A, const bf16 * B, float * C, int M, int N, int K,
               cudaStream_t stream = 0);
