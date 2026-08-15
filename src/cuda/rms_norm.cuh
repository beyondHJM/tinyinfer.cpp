#pragma once

#include "common.cuh"

// RMSNorm over the last dim. x/out: [rows, n], w: [n].
void rms_norm_f32(const float * x, const float * w, float eps, int n, int rows, float * out,
                  cudaStream_t stream = 0);

// Elementwise FP32 -> BF16 conversion.
void fp32_to_bf16_buf(const float * x, bf16 * y, size_t n, cudaStream_t stream = 0);

// out = a + b (elementwise, both [n])
void add_buf(const float * a, const float * b, float * out, size_t n, cudaStream_t stream = 0);

// Gather rows: out[sel] = x[idx[sel]], rows [count], each row [n].
void gather_rows(const float * x, const int * idx, int count, int n, float * out,
                 cudaStream_t stream = 0);
