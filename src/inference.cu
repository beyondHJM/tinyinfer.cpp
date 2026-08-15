#include "inference.h"

#include "cuda/attention.cuh"
#include "cuda/ffn.cuh"
#include "cuda/gemm.cuh"
#include "cuda/lm_head.cuh"
#include "cuda/rms_norm.cuh"

#include <cstring>

namespace {

__global__ void embed_kernel(const bf16 * __restrict__ embd, const int * __restrict__ tok,
                             int M, int n_embd, float * __restrict__ x) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int id = tok[m];
    const bf16 * row = embd + (size_t) id * n_embd;
    float * out = x + (size_t) m * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[i] = bf16_to_f32(row[i]);
    }
}

} // namespace

Inference::~Inference() {
    free_all();
}

void Inference::free_all() {
    auto fr = [](void * p) { if (p) cudaFree(p); };
    fr(x_); fr(xn_); fr(xb_); fr(qkv_); fr(qn_buf_); fr(attn_out_); fr(o_);
    fr(attn_out_b_); fr(ffn_b_);
    fr(gu_); fr(y_); fr(wgu_); fr(wqkv_); fr(x_last_); fr(x_last_b_); fr(logits_);
    fr(tok_dev_); fr(pos_dev_); fr(seq_dev_); fr(last_idx_dev_); fr(kv_);
    x_ = xn_ = nullptr; xb_ = nullptr; attn_out_b_ = nullptr; ffn_b_ = nullptr;
    qkv_ = nullptr; qn_buf_ = nullptr; attn_out_ = nullptr;
    o_ = nullptr; gu_ = nullptr; y_ = nullptr; wgu_ = nullptr; x_last_ = nullptr; x_last_b_ = nullptr;
    logits_ = nullptr; tok_dev_ = pos_dev_ = seq_dev_ = last_idx_dev_ = nullptr; kv_ = nullptr;
}

bool Inference::init(const Qwen3Model & model, int max_seq, int max_batch) {
    free_all();
    model_ = &model;
    max_seq_ = max_seq;
    max_batch_ = max_batch;
    max_tokens_ = max_seq * max_batch;

    const Qwen3Config & c = model.cfg;
    const int nq = c.n_head * c.head_dim;
    const int nk = c.n_head_kv * c.head_dim;

    auto alloc = [](auto & p, size_t bytes) {
        if (bytes == 0) return true;
        void * ptr = nullptr;
        cudaError_t e = cudaMalloc(&ptr, bytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(e));
            return false;
        }
        p = reinterpret_cast<decltype(p)>(ptr);
        return true;
    };

    bool ok = true;
    ok &= alloc(x_, (size_t) max_tokens_ * c.n_embd * 4);
    ok &= alloc(xn_, (size_t) max_tokens_ * c.n_embd * 4);
    ok &= alloc(xb_, (size_t) max_tokens_ * c.n_embd * 2);
    ok &= alloc(attn_out_b_, (size_t) max_tokens_ * nq * 2);
    ok &= alloc(ffn_b_, (size_t) max_tokens_ * c.n_ff * 2);
    ok &= alloc(qkv_, (size_t) max_tokens_ * (nq + 2 * nk) * 4);
    ok &= alloc(qn_buf_, (size_t) max_tokens_ * nq * 4);
    ok &= alloc(attn_out_, (size_t) max_tokens_ * nq * 4);
    ok &= alloc(o_, (size_t) max_tokens_ * c.n_embd * 4);
    ok &= alloc(gu_, (size_t) max_tokens_ * 2 * c.n_ff * 4);
    ok &= alloc(y_, (size_t) max_tokens_ * c.n_ff * 4);
    ok &= alloc(wgu_, (size_t) c.n_layer * 2 * c.n_ff * c.n_embd * 2);
    ok &= alloc(wqkv_, (size_t) c.n_layer * (nq + 2 * nk) * c.n_embd * 2);
    ok &= alloc(x_last_, (size_t) max_batch_ * c.n_embd * 4);
    ok &= alloc(x_last_b_, (size_t) max_batch_ * c.n_embd * 2);
    ok &= alloc(logits_, (size_t) max_batch_ * c.n_vocab * 4);
    ok &= alloc(tok_dev_, (size_t) max_tokens_ * 4);
    ok &= alloc(pos_dev_, (size_t) max_tokens_ * 4);
    ok &= alloc(seq_dev_, (size_t) max_tokens_ * 4);
    ok &= alloc(last_idx_dev_, (size_t) max_batch_ * 4);

    kv_seq_stride_ = (size_t) 2 * c.n_head_kv * c.head_dim * max_seq;
    kv_layer_stride_ = (size_t) max_batch * kv_seq_stride_;
    ok &= alloc(kv_, kv_layer_stride_ * c.n_layer * 2);
    if (!ok) {
        fprintf(stderr, "inference: out of memory during buffer allocation\n");
        return false;
    }

    // Combine gate and up into per-layer [2*n_ff, n_embd] weights.
    const size_t ff_bytes = (size_t) c.n_ff * c.n_embd * 2;
    for (int l = 0; l < c.n_layer; ++l) {
        bf16 * wl = wgu_ + (size_t) l * 2 * c.n_ff * c.n_embd;
        QWEN_CU_CHECK(cudaMemcpy(wl, model.layers[l].w_gate, ff_bytes, cudaMemcpyDeviceToDevice));
        QWEN_CU_CHECK(cudaMemcpy(wl + (size_t) c.n_ff * c.n_embd, model.layers[l].w_up, ff_bytes,
                                 cudaMemcpyDeviceToDevice));
        // Fused QKV weight: [nq + 2*nk, n_embd]
        bf16 * ql = wqkv_ + (size_t) l * (nq + 2 * nk) * c.n_embd;
        QWEN_CU_CHECK(cudaMemcpy(ql, model.layers[l].wq, (size_t) nq * c.n_embd * 2,
                                 cudaMemcpyDeviceToDevice));
        QWEN_CU_CHECK(cudaMemcpy(ql + (size_t) nq * c.n_embd, model.layers[l].wk,
                                 (size_t) nk * c.n_embd * 2, cudaMemcpyDeviceToDevice));
        QWEN_CU_CHECK(cudaMemcpy(ql + (size_t) (nq + nk) * c.n_embd, model.layers[l].wv,
                                 (size_t) nk * c.n_embd * 2, cudaMemcpyDeviceToDevice));
    }
    reset_kv();
    return true;
}

void Inference::reset_kv() {
    if (kv_) QWEN_CU_CHECK(cudaMemset(kv_, 0, kv_layer_stride_ * model_->cfg.n_layer * 2));
}

bool Inference::forward(const std::vector<int32_t> & tokens,
                        const std::vector<int> & positions,
                        const std::vector<int> & seq_ids,
                        int n_seqs,
                        std::vector<float> & last_logits) {
    const Qwen3Config & c = model_->cfg;
    const int M = (int) tokens.size();
    if (M == 0 || M > max_tokens_ || n_seqs > max_batch_) return false;

    QWEN_CU_CHECK(cudaMemcpy(tok_dev_, tokens.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    QWEN_CU_CHECK(cudaMemcpy(pos_dev_, positions.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    QWEN_CU_CHECK(cudaMemcpy(seq_dev_, seq_ids.data(), M * sizeof(int), cudaMemcpyHostToDevice));

    // last token index per sequence
    std::vector<int> last_idx(n_seqs, 0);
    for (int i = 0; i < M; ++i) last_idx[seq_ids[i]] = i;
    QWEN_CU_CHECK(cudaMemcpy(last_idx_dev_, last_idx.data(), n_seqs * sizeof(int),
                             cudaMemcpyHostToDevice));

    const int nq = c.n_head * c.head_dim;
    const int nk = c.n_head_kv * c.head_dim;

    embed_kernel<<<M, 256>>>(model_->tok_embd, tok_dev_, M, c.n_embd, x_);
    QWEN_CU_CHECK_LAST();

    for (int l = 0; l < c.n_layer; ++l) {
        const Qwen3Layer & L = model_->layers[l];

        // attention block
        rms_norm_f32(x_, L.attn_norm, c.norm_eps, c.n_embd, M, xn_);
        fp32_to_bf16_buf(xn_, xb_, (size_t) M * c.n_embd);
        gemm_bf16(xb_, wqkv_ + (size_t) l * (nq + 2 * nk) * c.n_embd, qkv_, M, nq + 2 * nk, c.n_embd);

        qk_norm_store(qkv_, qkv_ + nq, qkv_ + nq + nk, nq + 2 * nk, nq + 2 * nk, nq + 2 * nk,
                      L.q_norm, L.k_norm, c.norm_eps, c.rope_theta,
                      c.head_dim, c.n_head, c.n_head_kv,
                      pos_dev_, seq_dev_, M,
                      kv_ + l * kv_layer_stride_, (int) kv_seq_stride_, max_seq_,
                      qn_buf_);
        attention_compute(qn_buf_, pos_dev_, seq_dev_, M,
                          kv_ + l * kv_layer_stride_, (int) kv_seq_stride_, max_seq_,
                          c.head_dim, c.n_head, c.n_head_kv, attn_out_);
        fp32_to_bf16_buf(attn_out_, attn_out_b_, (size_t) M * nq);
        gemm_bf16(attn_out_b_, L.wo, o_, M, c.n_embd, nq);
        add_buf(x_, o_, x_, (size_t) M * c.n_embd);

        // FFN block
        rms_norm_f32(x_, L.ffn_norm, c.norm_eps, c.n_embd, M, xn_);
        fp32_to_bf16_buf(xn_, xb_, (size_t) M * c.n_embd);
        gemm_bf16(xb_, wgu_ + (size_t) l * 2 * c.n_ff * c.n_embd, gu_, M, 2 * c.n_ff, c.n_embd);
        swiglu_act(gu_, M, c.n_ff, y_);
        fp32_to_bf16_buf(y_, ffn_b_, (size_t) M * c.n_ff);
        gemm_bf16(ffn_b_, L.w_down, o_, M, c.n_embd, c.n_ff);
        add_buf(x_, o_, x_, (size_t) M * c.n_embd);
    }

    rms_norm_f32(x_, model_->output_norm, c.norm_eps, c.n_embd, M, xn_);
    gather_rows(xn_, last_idx_dev_, n_seqs, c.n_embd, x_last_);
    lm_head_forward(x_last_, n_seqs, model_->tok_embd, c.n_vocab, c.n_embd,
                    x_last_b_, logits_);

    last_logits.resize((size_t) n_seqs * c.n_vocab);
    QWEN_CU_CHECK(cudaMemcpy(last_logits.data(), logits_, last_logits.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    return true;
}
