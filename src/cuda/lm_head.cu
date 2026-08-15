#include "lm_head.cuh"
#include "gemm.cuh"
#include "rms_norm.cuh"

void lm_head_forward(const float * x, int M, const bf16 * w, int vocab, int n_embd,
                     bf16 * xb, float * logits, cudaStream_t stream) {
    if (M <= 0) return;
    fp32_to_bf16_buf(x, xb, (size_t) M * n_embd, stream);
    gemm_bf16(xb, w, logits, M, vocab, n_embd, stream);
}
