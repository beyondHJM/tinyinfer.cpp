#include "ffn.cuh"

#include <cmath>

namespace {

__global__ void swiglu_kernel(const float * __restrict__ gu, int M, int n_ff,
                              float * __restrict__ y) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) M * n_ff) return;
    const size_t m = i / n_ff;
    const size_t j = i % n_ff;
    const float gate = gu[m * 2 * n_ff + j];
    const float up = gu[m * 2 * n_ff + n_ff + j];
    y[i] = gate / (1.0f + expf(-gate)) * up;   // silu(gate) * up
}

} // namespace

void swiglu_act(const float * gu, int M, int n_ff, float * y, cudaStream_t stream) {
    if (M <= 0) return;
    const size_t n = (size_t) M * n_ff;
    const int threads = 256;
    const size_t blocks = (n + threads - 1) / threads;
    swiglu_kernel<<<(unsigned) blocks, threads, 0, stream>>>(gu, M, n_ff, y);
}
