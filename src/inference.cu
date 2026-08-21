#include "inference.h"

#include "cuda/attention.cuh"
#include "cuda/ffn.cuh"
#include "cuda/gemm.cuh"
#include "cuda/lm_head.cuh"
#include "cuda/rms_norm.cuh"
#include "cuda/sample.cuh"

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

void Inference::launch_forward_kernels(int M, int n_seqs, cudaStream_t stream) {
    const Qwen3Config & c = model_->cfg;
    const int nq = c.n_head * c.head_dim;
    const int nk = c.n_head_kv * c.head_dim;

    embed_kernel<<<M, 256, 0, stream>>>(model_->tok_embd, tok_dev_, M, c.n_embd, x_);

    for (int l = 0; l < c.n_layer; ++l) {
        const Qwen3Layer & L = model_->layers[l];

        // attention block
        rms_norm_f32(x_, L.attn_norm, c.norm_eps, c.n_embd, M, xn_, stream);
        fp32_to_bf16_buf(xn_, xb_, (size_t) M * c.n_embd, stream);
        gemm_bf16(xb_, wqkv_ + (size_t) l * (nq + 2 * nk) * c.n_embd, qkv_, M, nq + 2 * nk,
                  c.n_embd, stream);

        qk_norm_store(qkv_, qkv_ + nq, qkv_ + nq + nk, nq + 2 * nk, nq + 2 * nk, nq + 2 * nk,
                      L.q_norm, L.k_norm, c.norm_eps, c.rope_theta,
                      c.head_dim, c.n_head, c.n_head_kv,
                      pos_dev_, seq_dev_, M,
                      kv_ + l * kv_layer_stride_, (int) kv_seq_stride_, max_seq_,
                      qn_buf_, stream);
        attention_compute(qn_buf_, pos_dev_, seq_dev_, M,
                          kv_ + l * kv_layer_stride_, (int) kv_seq_stride_, max_seq_,
                          c.head_dim, c.n_head, c.n_head_kv, attn_out_, stream);
        fp32_to_bf16_buf(attn_out_, attn_out_b_, (size_t) M * nq, stream);
        gemm_bf16(attn_out_b_, L.wo, o_, M, c.n_embd, nq, stream);
        add_buf(x_, o_, x_, (size_t) M * c.n_embd, stream);

        // FFN block
        rms_norm_f32(x_, L.ffn_norm, c.norm_eps, c.n_embd, M, xn_, stream);
        fp32_to_bf16_buf(xn_, xb_, (size_t) M * c.n_embd, stream);
        gemm_bf16(xb_, wgu_ + (size_t) l * 2 * c.n_ff * c.n_embd, gu_, M, 2 * c.n_ff, c.n_embd,
                  stream);
        swiglu_act(gu_, M, c.n_ff, y_, stream);
        fp32_to_bf16_buf(y_, ffn_b_, (size_t) M * c.n_ff, stream);
        gemm_bf16(ffn_b_, L.w_down, o_, M, c.n_embd, c.n_ff, stream);
        add_buf(x_, o_, x_, (size_t) M * c.n_embd, stream);
    }

    rms_norm_f32(x_, model_->output_norm, c.norm_eps, c.n_embd, M, xn_, stream);
    gather_rows(xn_, last_idx_dev_, n_seqs, c.n_embd, x_last_, stream);
    lm_head_forward(x_last_, n_seqs, model_->tok_embd, c.n_vocab, c.n_embd,
                    x_last_b_, logits_, stream);
}

Inference::~Inference() {
    free_all();
}

void Inference::free_all() {
    for (auto & [m, g] : decode_graphs_) {
        if (g) cudaGraphExecDestroy(g);
    }
    decode_graphs_.clear();
    if (graph_stream_) cudaStreamDestroy(graph_stream_);
    graph_stream_ = nullptr;

    auto fr = [](void * p) { if (p) cudaFree(p); };
    fr(x_); fr(xn_); fr(xb_); fr(qkv_); fr(qn_buf_); fr(attn_out_); fr(o_);
    fr(attn_out_b_); fr(ffn_b_);
    fr(gu_); fr(y_); fr(wgu_); fr(wqkv_); fr(x_last_); fr(x_last_b_); fr(logits_);
    fr(tok_dev_); fr(pos_dev_); fr(seq_dev_); fr(last_idx_dev_); fr(kv_);
    fr(cand_vals_); fr(cand_idx_);
    x_ = xn_ = nullptr; xb_ = nullptr; attn_out_b_ = nullptr; ffn_b_ = nullptr;
    qkv_ = nullptr; qn_buf_ = nullptr; attn_out_ = nullptr;
    o_ = nullptr; gu_ = nullptr; y_ = nullptr; wgu_ = nullptr; x_last_ = nullptr; x_last_b_ = nullptr;
    logits_ = nullptr; tok_dev_ = pos_dev_ = seq_dev_ = last_idx_dev_ = nullptr; kv_ = nullptr;
    cand_vals_ = nullptr; cand_idx_ = nullptr;
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
    ok &= alloc(cand_vals_, (size_t) max_batch_ * SAM_CAND_PER_SEQ * 4);
    ok &= alloc(cand_idx_, (size_t) max_batch_ * SAM_CAND_PER_SEQ * 4);

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
                        std::vector<float> & last_logits,
                        std::vector<float> * cand_probs,
                        std::vector<int> * cand_idx,
                        std::vector<int> * greedy_ids) {
    const int M = (int) tokens.size();
    if (M == 0 || M > max_tokens_ || n_seqs > max_batch_) return false;
    const int n_vocab = model_->cfg.n_vocab;

    QWEN_CU_CHECK(cudaMemcpy(tok_dev_, tokens.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    QWEN_CU_CHECK(cudaMemcpy(pos_dev_, positions.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    QWEN_CU_CHECK(cudaMemcpy(seq_dev_, seq_ids.data(), M * sizeof(int), cudaMemcpyHostToDevice));

    // last token index per sequence
    std::vector<int> last_idx(n_seqs, 0);
    for (int i = 0; i < M; ++i) last_idx[seq_ids[i]] = i;
    QWEN_CU_CHECK(cudaMemcpy(last_idx_dev_, last_idx.data(), n_seqs * sizeof(int),
                             cudaMemcpyHostToDevice));

    // CUDA Graph path: only used for decode steps. Decode has at most one
    // token per sequence (M <= n_seqs); prefill has M > n_seqs (multiple tokens
    // per sequence), which is never captured. The input buffers above are
    // updated with blocking copies on the default stream, so they are
    // guaranteed to be visible before the graph replays on graph_stream_
    // (blocking copies synchronize the whole device).
    if (use_cuda_graph_ && M <= n_seqs) {
        cudaGraphExec_t g = get_or_capture_decode_graph(M, n_seqs);
        if (g) {
            QWEN_CU_CHECK(cudaGraphLaunch(g, graph_stream_));
            if (greedy_ids) {
                sample_argmax(logits_, n_seqs, n_vocab, cand_idx_, graph_stream_);
                greedy_ids->resize(n_seqs);
                QWEN_CU_CHECK(cudaMemcpyAsync(greedy_ids->data(), cand_idx_,
                                              greedy_ids->size() * sizeof(int),
                                              cudaMemcpyDeviceToHost, graph_stream_));
                QWEN_CU_CHECK(cudaStreamSynchronize(graph_stream_));
                return true;
            }
            QWEN_CU_CHECK(cudaStreamSynchronize(graph_stream_));
            if (cand_probs && cand_idx) {
                sample_topk_probs(logits_, n_seqs, n_vocab, cand_vals_, cand_idx_, 0);
                cand_probs->resize((size_t) n_seqs * SAM_CAND_PER_SEQ);
                cand_idx->resize((size_t) n_seqs * SAM_CAND_PER_SEQ);
                QWEN_CU_CHECK(cudaMemcpy(cand_probs->data(), cand_vals_,
                                         cand_probs->size() * sizeof(float),
                                         cudaMemcpyDeviceToHost));
                QWEN_CU_CHECK(cudaMemcpy(cand_idx->data(), cand_idx_,
                                         cand_idx->size() * sizeof(int),
                                         cudaMemcpyDeviceToHost));
            } else {
                last_logits.resize((size_t) n_seqs * n_vocab);
                QWEN_CU_CHECK(cudaMemcpy(last_logits.data(), logits_,
                                         last_logits.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost));
            }
            return true;
        }
        fprintf(stderr, "inference: cuda graph unavailable for M=%d, falling back to eager\n", M);
    }

    launch_forward_kernels(M, n_seqs, 0);
    QWEN_CU_CHECK_LAST();

    if (greedy_ids) {
        sample_argmax(logits_, n_seqs, n_vocab, cand_idx_, 0);
        greedy_ids->resize(n_seqs);
        QWEN_CU_CHECK(cudaMemcpy(greedy_ids->data(), cand_idx_,
                                 greedy_ids->size() * sizeof(int), cudaMemcpyDeviceToHost));
    } else if (cand_probs && cand_idx) {
        sample_topk_probs(logits_, n_seqs, n_vocab, cand_vals_, cand_idx_, 0);
        cand_probs->resize((size_t) n_seqs * SAM_CAND_PER_SEQ);
        cand_idx->resize((size_t) n_seqs * SAM_CAND_PER_SEQ);
        QWEN_CU_CHECK(cudaMemcpy(cand_probs->data(), cand_vals_,
                                 cand_probs->size() * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        QWEN_CU_CHECK(cudaMemcpy(cand_idx->data(), cand_idx_,
                                 cand_idx->size() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    } else {
        last_logits.resize((size_t) n_seqs * n_vocab);
        QWEN_CU_CHECK(cudaMemcpy(last_logits.data(), logits_, last_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    }
    return true;
}

cudaGraphExec_t Inference::get_or_capture_decode_graph(int M, int n_seqs) {
    for (auto & [m, g] : decode_graphs_) {
        if (m == M) return g;
    }
    // Cap the number of distinct graphs; beyond this we fall back to eager.
    if (M <= 0 || M > max_batch_ || (int) decode_graphs_.size() >= max_batch_) return nullptr;
    if (!graph_stream_) QWEN_CU_CHECK(cudaStreamCreate(&graph_stream_));

    cudaGraph_t graph = nullptr;
    QWEN_CU_CHECK(cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeThreadLocal));
    launch_forward_kernels(M, n_seqs, graph_stream_);
    QWEN_CU_CHECK(cudaStreamEndCapture(graph_stream_, &graph));

    cudaGraphExec_t exec = nullptr;
#if CUDART_VERSION >= 12000
    QWEN_CU_CHECK(cudaGraphInstantiate(&exec, graph, 0));
#else
    QWEN_CU_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
#endif

    size_t num_nodes = 0;
    cudaGraphGetNodes(graph, nullptr, &num_nodes);
    QWEN_CU_CHECK(cudaGraphDestroy(graph));
    fprintf(stderr, "inference: captured decode cuda graph M=%d (%zu nodes)\n", M, num_nodes);
    decode_graphs_.emplace_back(M, exec);

    if (getenv("QWEN_GRAPH_VERIFY")) verify_graph_vs_eager(M, n_seqs, exec);
    return exec;
}

void Inference::verify_graph_vs_eager(int M, int n_seqs, cudaGraphExec_t exec) {
    // Snapshot the KV cache, run one eager forward and one graph replay on the
    // same KV state, and compare the resulting logits numerically.
    const size_t kv_bytes = kv_layer_stride_ * model_->cfg.n_layer * 2;
    void * snap = nullptr;
    QWEN_CU_CHECK(cudaMalloc(&snap, kv_bytes));
    QWEN_CU_CHECK(cudaMemcpy(snap, kv_, kv_bytes, cudaMemcpyDeviceToDevice));

    launch_forward_kernels(M, n_seqs, 0);
    QWEN_CU_CHECK(cudaDeviceSynchronize());
    std::vector<float> logits_eager((size_t) n_seqs * model_->cfg.n_vocab);
    QWEN_CU_CHECK(cudaMemcpy(logits_eager.data(), logits_, logits_eager.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

    QWEN_CU_CHECK(cudaMemcpy(kv_, snap, kv_bytes, cudaMemcpyDeviceToDevice));
    QWEN_CU_CHECK(cudaGraphLaunch(exec, graph_stream_));
    QWEN_CU_CHECK(cudaStreamSynchronize(graph_stream_));
    std::vector<float> logits_graph((size_t) n_seqs * model_->cfg.n_vocab);
    QWEN_CU_CHECK(cudaMemcpy(logits_graph.data(), logits_, logits_graph.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

    QWEN_CU_CHECK(cudaFree(snap));

    float max_diff = 0.0f;
    int n_diff = 0;
    for (size_t i = 0; i < logits_eager.size(); ++i) {
        float d = fabsf(logits_eager[i] - logits_graph[i]);
        if (d > max_diff) max_diff = d;
        if (d != 0.0f) n_diff++;
    }
    int argmax_mismatch = 0;
    for (int s = 0; s < n_seqs; ++s) {
        const float * a = &logits_eager[(size_t) s * model_->cfg.n_vocab];
        const float * b = &logits_graph[(size_t) s * model_->cfg.n_vocab];
        int ia = 0, ib = 0;
        for (int i = 1; i < model_->cfg.n_vocab; ++i) {
            if (a[i] > a[ia]) ia = i;
            if (b[i] > b[ib]) ib = i;
        }
        if (ia != ib) argmax_mismatch++;
    }
    fprintf(stderr, "graph verify M=%d: max |logits diff| = %.6g, differing elems = %d/%zu, "
                    "argmax mismatches = %d/%d\n",
            M, max_diff, n_diff, logits_eager.size(), argmax_mismatch, n_seqs);
}
