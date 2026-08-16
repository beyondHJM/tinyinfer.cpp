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

// FlashDecoding-style attention for decode.
//
// Long contexts were previously scanned by a single block serially, so
// per-step cost grew linearly with position and only 16 blocks (one per head)
// were active. This kernel:
//   * splits the KV range into FLASH_NSPLIT=16 sub-ranges, each handled by a
//     16-lane half-warp group inside the block (block = 16 groups x 16 lanes);
//   * every group runs an online softmax (running max/sum with rescaling), so
//     K and V are each read exactly once (vectorized 16B uint4 loads, 8 dims
//     per lane);
//   * the 16 partial results are combined inside the block with the standard
//     flash rescale rule, yielding the exact same distribution as a full
//     softmax.
// The math is the classic flash-attention single-pass algorithm; the
// split-and-merge over the sequence dimension is the flash-decoding idea.
constexpr int FLASH_NSPLIT = 16;   // half-warp groups per block (block=256)
constexpr int FLASH_LANES = 16;    // lanes per group (8 dims each)

__global__ void __launch_bounds__(FLASH_NSPLIT * FLASH_LANES)
attention_compute_kernel(
        const float * __restrict__ q,
        const int * __restrict__ positions, const int * __restrict__ seq_ids, int M,
        const bf16 * __restrict__ kv_cache, int kv_stride, int max_seq,
        int head_dim, int n_head, int n_kv,
        float * __restrict__ out) {
    __shared__ float sout[FLASH_NSPLIT][128];   // partial outputs (dims)
    __shared__ float sm[FLASH_NSPLIT], sl[FLASH_NSPLIT];
    __shared__ float msh;

    const int m = blockIdx.x;
    const int h = blockIdx.y;
    if (m >= M || h >= n_head) return;
    const int tid = threadIdx.x;
    const int group = tid >> 4;      // 0..15
    const int lane = tid & 15;       // 0..15
    const int pos = positions[m];
    const int seq = seq_ids[m];

    const int q_per_kv = n_head / n_kv;
    const float rsqrt = rsqrtf((float) head_dim);
    const bf16 * cache = kv_cache + (size_t) seq * kv_stride;
    const int kvh = h / q_per_kv;
    const bf16 * kpos = cache + (size_t) kvh * max_seq * head_dim;
    const bf16 * vpos = cache + (size_t) n_kv * max_seq * head_dim
                              + (size_t) kvh * max_seq * head_dim;

    // This lane's 8 dims and its q slice.
    const int d0 = lane * 8;
    const float * qh = q + (size_t) m * (n_head * head_dim) + h * head_dim;
    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = qh[d0 + i];

    // Online softmax over this group's positions: p = group, group+16, ...
    float m_ = -INFINITY, l_ = 0.0f;
    float o8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned shmask = (tid & 16) ? 0xffff0000u : 0xffffu;

    for (int p = group; p <= pos; p += FLASH_NSPLIT) {
        const uint4 k4 = reinterpret_cast<const uint4 *>(kpos + (size_t) p * head_dim)[lane];
        const uint4 v4 = reinterpret_cast<const uint4 *>(vpos + (size_t) p * head_dim)[lane];
        const bf16 * kb = reinterpret_cast<const bf16 *>(&k4);
        const bf16 * vb = reinterpret_cast<const bf16 *>(&v4);
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) s += qv[i] * bf16_to_f32(kb[i]);
        // Half-warp reduction (results beyond lane 0 are garbage but unused).
        float sr = s;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) sr += __shfl_down_sync(shmask, sr, o);
        sr = __shfl_sync(shmask, sr, tid & 16);
        const float sc = sr * rsqrt;
        const float m_new = fmaxf(m_, sc);
        const float alpha = __expf(m_ - m_new);
        const float pp = __expf(sc - m_new);
        l_ = l_ * alpha + pp;
#pragma unroll
        for (int i = 0; i < 8; ++i) o8[i] = o8[i] * alpha + pp * bf16_to_f32(vb[i]);
        m_ = m_new;
    }

    // Publish this group's partial result.
#pragma unroll
    for (int i = 0; i < 8; ++i) sout[group][d0 + i] = o8[i];
    if (lane == 0) { sm[group] = m_; sl[group] = l_; }
    __syncthreads();

    // Combine the 16 partials (flash rescale rule).
    if (lane == 0) {
        float mm = -INFINITY;
        for (int g = 0; g < FLASH_NSPLIT; ++g) mm = fmaxf(mm, sm[g]);
        msh = mm;
    }
    __syncthreads();
    const float m_max = msh;

    float lf = 0.0f;
    float of[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int g = 0; g < FLASH_NSPLIT; ++g) {
        const float w = __expf(sm[g] - m_max);
        lf += sl[g] * w;
#pragma unroll
        for (int i = 0; i < 8; ++i) of[i] += sout[g][d0 + i] * w;
    }
    const float inv = 1.0f / lf;
    float * oh = out + (size_t) m * (n_head * head_dim) + h * head_dim;
#pragma unroll
    for (int i = 0; i < 8; ++i) oh[d0 + i] = of[i] * inv;
}

// ---------------------------------------------------------------------------
// Prefill attention: many tokens at once (M > n_seqs).
//
// Measured on this engine, the flash-decoding kernel above is already the best
// CUDA-core attention shape for prefill as well: one block per token with 16
// half-warp groups splitting the position range gives the finest per-token
// parallelism, and profiling shows attention is only ~12% of prefill time
// (GEMMs are ~74%). A separate prefill kernel would only pay off with tensor
// cores (flash-attention-2 style), which is a larger undertaking; it is kept
// here as a reference/experiment entry point.
//
// The variant below assigns one 16-thread group per token without sequence
// splitting (simplest possible): each group scans its full causal prefix with
// an online softmax and vectorized reads. It is *not* used by the engine
// because long positions become serial per group; retained for experiments.
// ---------------------------------------------------------------------------
constexpr int PREF_BM = 16;    // tokens per block (one per group)
constexpr int PREF_LANES = 16; // lanes per group (8 dims each)

__global__ void __launch_bounds__(PREF_BM * PREF_LANES)
prefill_attention_kernel(
        const float * __restrict__ q,
        const int * __restrict__ positions, const int * __restrict__ seq_ids, int M,
        const bf16 * __restrict__ kv_cache, int kv_stride, int max_seq,
        int head_dim, int n_head, int n_kv,
        float * __restrict__ out) {
    const int m0 = blockIdx.x * PREF_BM;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int group = tid >> 4;   // which token in the block
    const int lane = tid & 15;    // 0..15, dims lane*8..lane*8+7
    const int m = m0 + group;
    const bool active = m < M;
    const int pos = active ? positions[m] : 0;
    const int seq = active ? seq_ids[m] : 0;

    const int nq = n_head * head_dim;
    const int kvh = h / (n_head / n_kv);
    const int d0 = lane * 8;
    const float rsqrt = rsqrtf((float) head_dim);
    const unsigned shmask = (tid & 16) ? 0xffff0000u : 0xffffu;

    float qv[8];
    if (active) {
        const float * qh = q + (size_t) m * nq + h * head_dim;
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = qh[d0 + i];
    }
    const bf16 * cache = kv_cache + (size_t) seq * kv_stride;
    const bf16 * kpos = cache + (size_t) kvh * max_seq * head_dim;
    const bf16 * vpos = cache + (size_t) (n_kv + kvh) * max_seq * head_dim;

    float m_ = -INFINITY, l_ = 0.0f;
    float o8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (active) {
        for (int p = 0; p <= pos; ++p) {
            const uint4 k4 = reinterpret_cast<const uint4 *>(kpos + (size_t) p * head_dim)[lane];
            const uint4 v4 = reinterpret_cast<const uint4 *>(vpos + (size_t) p * head_dim)[lane];
            const bf16 * kb = reinterpret_cast<const bf16 *>(&k4);
            const bf16 * vb = reinterpret_cast<const bf16 *>(&v4);
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) s += qv[i] * bf16_to_f32(kb[i]);
            float sr = s;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sr += __shfl_down_sync(shmask, sr, o);
            sr = __shfl_sync(shmask, sr, tid & 16);
            const float sc = sr * rsqrt;
            const float m_new = fmaxf(m_, sc);
            const float alpha = __expf(m_ - m_new);
            const float pp = __expf(sc - m_new);
            l_ = l_ * alpha + pp;
#pragma unroll
            for (int i = 0; i < 8; ++i) o8[i] = o8[i] * alpha + pp * bf16_to_f32(vb[i]);
            m_ = m_new;
        }
    }
    if (active) {
        float * oh = out + (size_t) m * nq + h * head_dim;
        const float inv = 1.0f / l_;
#pragma unroll
        for (int i = 0; i < 8; ++i) oh[d0 + i] = o8[i] * inv;
    }
}

} // namespace

void prefill_attention_compute(const float * q, const int * positions, const int * seq_ids,
                               int M, const bf16 * kv_cache, int kv_stride, int max_seq,
                               int head_dim, int n_head, int n_kv,
                               float * out, cudaStream_t stream) {
    if (M <= 0) return;
    if (head_dim != 128) {
        fprintf(stderr, "attention: prefill kernel requires head_dim == 128 (got %d)\n", head_dim);
        return;
    }
    dim3 grid((M + PREF_BM - 1) / PREF_BM, n_head);
    prefill_attention_kernel<<<grid, PREF_BM * PREF_LANES, 0, stream>>>(
        q, positions, seq_ids, M, kv_cache, kv_stride, max_seq, head_dim, n_head, n_kv, out);
}

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
    if (head_dim != 128) {
        fprintf(stderr, "attention: flash kernel requires head_dim == 128 (got %d)\n", head_dim);
        return;
    }
    dim3 grid(M, n_head);
    attention_compute_kernel<<<grid, FLASH_NSPLIT * FLASH_LANES, 0, stream>>>(
        q, positions, seq_ids, M, kv_cache, kv_stride, max_seq, head_dim, n_head, n_kv, out);
}
