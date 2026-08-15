#include "attention.cuh"

#include <cmath>

namespace {

constexpr int THREADS = 128;
constexpr int MAX_NQ = 2048;   // Qwen3-0.6B: n_head*head_dim = 2048
constexpr int MAX_NK = 1024;   // n_head_kv*head_dim = 1024
constexpr int MAX_HD = 256;

__global__ void qk_norm_store_kernel(
        const float * __restrict__ q, const float * __restrict__ k,
        const float * __restrict__ v,
        int q_stride, int k_stride, int v_stride,
        const float * __restrict__ q_norm, const float * __restrict__ k_norm,
        float norm_eps, float rope_theta,
        int head_dim, int n_head, int n_kv,
        const int * __restrict__ positions, const int * __restrict__ seq_ids, int M,
        const bf16 * __restrict__ kv_cache, int kv_stride, int max_seq,
        float * __restrict__ q_out) {
    __shared__ float qs[MAX_NQ];
    __shared__ float ks[MAX_NK];
    __shared__ float vs[MAX_NK];
    __shared__ float qn[MAX_HD];
    __shared__ float kn[MAX_HD];

    const int m = blockIdx.x;
    const int h = blockIdx.y;
    if (m >= M || h >= n_head) return;
    const int tid = threadIdx.x;
    const int pos = positions[m];
    const int seq = seq_ids[m];
    const int nq = n_head * head_dim;
    const int nk = n_kv * head_dim;

    const int q_per_kv = n_head / n_kv;
    const float theta_scale = powf(rope_theta, -2.0f / (float) head_dim);
    const bf16 * cache = kv_cache + (size_t) seq * kv_stride;

    const int kvh = h / q_per_kv;

    // Load this head's Q/K/V slices.
    for (int d = tid; d < head_dim; d += THREADS) {
        qs[d] = q[(size_t) m * q_stride + h * head_dim + d];
        ks[d] = k[(size_t) m * k_stride + kvh * head_dim + d];
        vs[d] = v[(size_t) m * v_stride + kvh * head_dim + d];
    }
    __syncthreads();

    // Q RMSNorm + RoPE (norm first, then rope, matching qwen3.cpp)
    float sq = 0.0f;
    for (int d = tid; d < head_dim; d += THREADS) sq += qs[d] * qs[d];
    float rms_q = rsqrtf(block_reduce_sum<0>(sq) / (float) head_dim + norm_eps);
    for (int d = tid; d < head_dim; d += THREADS) qn[d] = qs[d] * rms_q * q_norm[d];

    float sk = 0.0f;
    for (int d = tid; d < head_dim; d += THREADS) sk += ks[d] * ks[d];
    float rms_k = rsqrtf(block_reduce_sum<1>(sk) / (float) head_dim + norm_eps);
    for (int d = tid; d < head_dim; d += THREADS) kn[d] = ks[d] * rms_k * k_norm[d];

    __syncthreads();   // qn/kn must be visible to the RoPE threads

    if (tid < head_dim / 2) {
        const float angle = pos * powf(theta_scale, (float) tid);
        const float c = cosf(angle);
        const float s = sinf(angle);
        float x0 = qn[tid], x1 = qn[tid + head_dim / 2];
        qn[tid]              = x0 * c - x1 * s;
        qn[tid + head_dim/2] = x0 * s + x1 * c;
        x0 = kn[tid]; x1 = kn[tid + head_dim / 2];
        kn[tid]              = x0 * c - x1 * s;
        kn[tid + head_dim/2] = x0 * s + x1 * c;
    }
    __syncthreads();

    const bf16 * kpos = cache + (size_t) kvh * max_seq * head_dim + (size_t) pos * head_dim;
    const bf16 * vpos = cache + (size_t) n_kv * max_seq * head_dim
                              + (size_t) kvh * max_seq * head_dim
                              + (size_t) pos * head_dim;
    float * qoh = q_out + (size_t) m * nq + h * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        ((bf16 *) kpos)[d] = f32_to_bf16(kn[d]);
        ((bf16 *) vpos)[d] = f32_to_bf16(vs[d]);
        qoh[d] = qn[d];
    }
}

__global__ void attention_compute_kernel(
        const float * __restrict__ q,
        const int * __restrict__ positions, const int * __restrict__ seq_ids, int M,
        const bf16 * __restrict__ kv_cache, int kv_stride, int max_seq,
        int head_dim, int n_head, int n_kv,
        float * __restrict__ out) {
    extern __shared__ float scores[];
    __shared__ float qs[MAX_NQ];

    const int m = blockIdx.x;
    const int h = blockIdx.y;
    if (m >= M || h >= n_head) return;
    const int tid = threadIdx.x;
    const int pos = positions[m];
    const int seq = seq_ids[m];

    const int q_per_kv = n_head / n_kv;
    const float rsqrt = rsqrtf((float) head_dim);
    const bf16 * cache = kv_cache + (size_t) seq * kv_stride;

    const int kvh = h / q_per_kv;
    const float * qh = q + (size_t) m * (n_head * head_dim) + h * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) qs[d] = qh[d];
    __syncthreads();

    const bf16 * kpos = cache + (size_t) kvh * max_seq * head_dim;
    const bf16 * vpos = cache + (size_t) n_kv * max_seq * head_dim
                              + (size_t) kvh * max_seq * head_dim;

    float smax = -INFINITY;
    for (int p = tid; p <= pos; p += THREADS) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) s += qs[d] * bf16_to_f32(kpos[(size_t) p * head_dim + d]);
        scores[p] = s * rsqrt;
        smax = fmaxf(smax, scores[p]);
    }
    smax = block_reduce_max<0>(smax);

    float ssum = 0.0f;
    for (int p = tid; p <= pos; p += THREADS) {
        scores[p] = expf(scores[p] - smax);
        ssum += scores[p];
    }
    ssum = block_reduce_sum<1>(ssum);

    float * oh = out + (size_t) m * (n_head * head_dim) + h * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        float acc = 0.0f;
        for (int p = 0; p <= pos; ++p) {
            acc += scores[p] * bf16_to_f32(vpos[(size_t) p * head_dim + d]);
        }
        oh[d] = acc / ssum;
    }
}

} // namespace

void qk_norm_store(const float * q, const float * k, const float * v,
                   int q_stride, int k_stride, int v_stride,
                   const float * q_norm, const float * k_norm,
                   float norm_eps, float rope_theta,
                   int head_dim, int n_head, int n_kv,
                   const int * positions, const int * seq_ids, int M,
                   bf16 * kv_cache, int kv_stride, int max_seq,
                   float * q_out, cudaStream_t stream) {
    if (M <= 0) return;
    dim3 grid(M, n_head);
    qk_norm_store_kernel<<<grid, THREADS, 0, stream>>>(
        q, k, v, q_stride, k_stride, v_stride, q_norm, k_norm, norm_eps, rope_theta, head_dim, n_head, n_kv,
        positions, seq_ids, M, kv_cache, kv_stride, max_seq, q_out);
}

void attention_compute(const float * q, const int * positions, const int * seq_ids, int M,
                       const bf16 * kv_cache, int kv_stride, int max_seq,
                       int head_dim, int n_head, int n_kv,
                       float * out, cudaStream_t stream) {
    if (M <= 0) return;
    const size_t shmem = sizeof(float) * (size_t) max_seq;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(attention_compute_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr_set = true;
    }
    dim3 grid(M, n_head);
    attention_compute_kernel<<<grid, THREADS, shmem, stream>>>(
        q, positions, seq_ids, M, kv_cache, kv_stride, max_seq, head_dim, n_head, n_kv, out);
}
