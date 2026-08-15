#include "rms_norm.cuh"

namespace {

// One block per row. Block size 256.
__global__ void rms_norm_kernel(const float * __restrict__ x,
                                const float * __restrict__ w,
                                float eps, int n, int rows,
                                float * __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float * xr = x + (size_t) row * n;
    float * orow = out + (size_t) row * n;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    float mean = block_reduce_sum<0>(sum) / (float) n;
    float rms = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * rms * w[i];
    }
}

__global__ void fp32_to_bf16_kernel(const float * __restrict__ x, bf16 * __restrict__ y,
                                    size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f32_to_bf16(x[i]);
}

__global__ void add_kernel(const float * __restrict__ a, const float * __restrict__ b,
                           float * __restrict__ out, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void gather_rows_kernel(const float * __restrict__ x, const int * __restrict__ idx,
                                   int count, int n, float * __restrict__ out) {
    int r = blockIdx.x;
    if (r >= count) return;
    const int src = idx[r];
    const float * s = x + (size_t) src * n;
    float * d = out + (size_t) r * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

} // namespace

void rms_norm_f32(const float * x, const float * w, float eps, int n, int rows, float * out,
                  cudaStream_t stream) {
    if (rows <= 0) return;
    rms_norm_kernel<<<rows, 256, 0, stream>>>(x, w, eps, n, rows, out);
}

void fp32_to_bf16_buf(const float * x, bf16 * y, size_t n, cudaStream_t stream) {
    const int threads = 256;
    const size_t blocks = (n + threads - 1) / threads;
    fp32_to_bf16_kernel<<<(unsigned) blocks, threads, 0, stream>>>(x, y, n);
}

void add_buf(const float * a, const float * b, float * out, size_t n, cudaStream_t stream) {
    const int threads = 256;
    const size_t blocks = (n + threads - 1) / threads;
    add_kernel<<<(unsigned) blocks, threads, 0, stream>>>(a, b, out, n);
}

void gather_rows(const float * x, const int * idx, int count, int n, float * out,
                 cudaStream_t stream) {
    gather_rows_kernel<<<count, 256, 0, stream>>>(x, idx, count, n, out);
}
