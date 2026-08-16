#include "sample.cuh"

namespace {

__global__ void __launch_bounds__(SAM_THREADS)
topk_probs_kernel(const float * __restrict__ logits, int n_vocab,
                  float * __restrict__ cand_vals, int * __restrict__ cand_idx) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const float * row = logits + (size_t) m * n_vocab;

    // Pass 1: block maximum (softmax stability).
    float mx = -INFINITY;
    for (int i = tid; i < n_vocab; i += SAM_THREADS) mx = fmaxf(mx, row[i]);
    mx = block_reduce_max<0>(mx);

    // Pass 2: partition function sum(exp(x - max)).
    float s = 0.0f;
    for (int i = tid; i < n_vocab; i += SAM_THREADS) s += __expf(row[i] - mx);
    s = block_reduce_sum<1>(s);
    const float inv = 1.0f / fmaxf(s, 1e-30f);

    // Pass 3: per-thread local top-16 (insertion into a register array).
    float lv[SAM_TOP_PER_THREAD];
    int  li[SAM_TOP_PER_THREAD];
#pragma unroll
    for (int k = 0; k < SAM_TOP_PER_THREAD; ++k) { lv[k] = -1.0f; li[k] = -1; }
    for (int i = tid; i < n_vocab; i += SAM_THREADS) {
        const float p = __expf(row[i] - mx) * inv;
#pragma unroll
        for (int k = 0; k < SAM_TOP_PER_THREAD; ++k) {
            if (p > lv[k]) {
#pragma unroll
                for (int j = SAM_TOP_PER_THREAD - 1; j > k; --j) {
                    lv[j] = lv[j - 1];
                    li[j] = li[j - 1];
                }
                lv[k] = p;
                li[k] = i;
                break;
            }
        }
    }

    float * out_v = cand_vals + (size_t) m * SAM_CAND_PER_SEQ + tid * SAM_TOP_PER_THREAD;
    int   * out_i = cand_idx  + (size_t) m * SAM_CAND_PER_SEQ + tid * SAM_TOP_PER_THREAD;
#pragma unroll
    for (int k = 0; k < SAM_TOP_PER_THREAD; ++k) {
        out_v[k] = lv[k];
        out_i[k] = li[k];
    }
}

} // namespace

void sample_topk_probs(const float * logits, int M, int n_vocab,
                       float * cand_vals, int * cand_idx, cudaStream_t stream) {
    if (M <= 0 || n_vocab <= 0) return;
    topk_probs_kernel<<<M, SAM_THREADS, 0, stream>>>(logits, n_vocab, cand_vals, cand_idx);
}
