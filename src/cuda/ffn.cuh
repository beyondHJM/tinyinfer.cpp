#pragma once

#include "common.cuh"

// y[M, n_ff] = silu(gate[M, n_ff]) * up[M, n_ff], reading from gu[M, 2*n_ff].
// gu and y must NOT alias (in-place would race across batch rows).
void swiglu_act(const float * gu, int M, int n_ff, float * y, cudaStream_t stream = 0);
