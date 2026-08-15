#include "rope.cuh"

#include <cmath>

namespace {

// One thread per (row, head, pair). Pair k rotates (x[k], x[k + head_dim/2]).
__global__ void rope_neox_kernel(const float * __restrict__ x,
                                 const int * __restrict__ positions,
                                 int rows, int n_heads, int head_dim,
                                 float theta_scale,
                                 float * __restrict__ out) {
    const int k = blockIdx.x;                       // pair index in [0, head_dim/2)
    const int mh = blockIdx.y;                      // row * n_heads + head
    const int m = mh / n_heads;
    const int h = mh % n_heads;
    if (k >= head_dim / 2 || m >= rows) return;

    const int pos = positions[m];
    const float angle = pos * powf(theta_scale, (float) k);
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);

    const size_t base = (size_t) m * (n_heads * head_dim) + (size_t) h * head_dim;
    const float x0 = x[base + k];
    const float x1 = x[base + k + head_dim / 2];
    out[base + k]              = x0 * cos_a - x1 * sin_a;
    out[base + k + head_dim/2] = x0 * sin_a + x1 * cos_a;
}

} // namespace

void rope_neox(const float * x, const int * positions, int rows, int n_heads, int head_dim,
               float theta, float * out, cudaStream_t stream) {
    if (rows <= 0) return;
    const float theta_scale = powf(theta, -2.0f / (float) head_dim);
    dim3 grid(head_dim / 2, rows * n_heads);
    rope_neox_kernel<<<grid, 1, 0, stream>>>(x, positions, rows, n_heads, head_dim,
                                             theta_scale, out);
}
