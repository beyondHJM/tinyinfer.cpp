// Self-test: validate each CUDA operator against a CPU reference.
// Enabled with QWEN_SELFTEST=1. Returns nonzero on failure.

#include "cuda/attention.cuh"
#include "cuda/common.cuh"
#include "cuda/gemm.cuh"
#include "cuda/rms_norm.cuh"
#include "cuda/rope.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void report(const char * name, bool ok, float maxdiff, float tol) {
    printf("[selftest] %-14s %s  maxdiff=%.6g (tol=%.3g)\n",
           name, ok ? "PASS" : "FAIL", maxdiff, tol);
    if (!ok) g_fail++;
}

std::vector<bf16> random_bf16(size_t n, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bf16> v(n);
    for (auto & x : v) x = __float2bfloat16(dist(rng));
    return v;
}

void test_gemm() {
    std::mt19937 rng(1);
    const int M = 5, N = 261, K = 1024;
    auto A = random_bf16(M * K, rng);
    auto B = random_bf16(N * K, rng);

    bf16 * da, * db;
    float * dc;
    cudaMalloc(&da, A.size() * 2);
    cudaMalloc(&db, B.size() * 2);
    cudaMalloc(&dc, (size_t) M * N * 4);
    cudaMemcpy(da, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
    gemm_bf16(da, db, dc, M, N, K);
    std::vector<float> ours(M * N);
    cudaMemcpy(ours.data(), dc, ours.size() * 4, cudaMemcpyDeviceToHost);

    std::vector<float> ref(M * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                s += __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
            }
            ref[m * N + n] = (float) s;
        }
    }
    float mx = 0;
    size_t argmx = 0;
    int n_bad = 0;
    for (size_t i = 0; i < ours.size(); ++i) {
        float d = fabsf(ours[i] - ref[i]);
        if (d > 1.0f) n_bad++;
        if (d > mx) { mx = d; argmx = i; }
    }
    printf("[selftest] gemm detail: maxdiff=%.3f at (%d,%d) ours=%.3f ref=%.3f n_bad(>1)=%d\n",
           mx, (int)(argmx / N), (int)(argmx % N), ours[argmx], ref[argmx], n_bad);
    report("gemm", mx < 0.05f * K, mx, 0.05f * K);
    cudaFree(da); cudaFree(db); cudaFree(dc);
}

void test_rms_norm() {
    std::mt19937 rng(2);
    const int n = 1024, rows = 3;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(rows * n), w(n), ref(rows * n);
    for (auto & v : x) v = dist(rng);
    for (auto & v : w) v = dist(rng);
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += (double) x[r * n + i] * x[r * n + i];
        float rms = 1.0f / sqrtf((float) (ss / n) + 1e-6f);
        for (int i = 0; i < n; ++i) ref[r * n + i] = x[r * n + i] * rms * w[i];
    }
    float * dx, * dw, * dout;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&dw, w.size() * 4);
    cudaMalloc(&dout, x.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    rms_norm_f32(dx, dw, 1e-6f, n, rows, dout);
    std::vector<float> ours(x.size());
    cudaMemcpy(ours.data(), dout, ours.size() * 4, cudaMemcpyDeviceToHost);
    float mx = 0;
    for (size_t i = 0; i < ours.size(); ++i) mx = fmaxf(mx, fabsf(ours[i] - ref[i]));
    report("rms_norm", mx < 1e-4f, mx, 1e-4f);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

void test_rope() {
    std::mt19937 rng(3);
    const int rows = 2, heads = 4, hd = 128, theta = 1000000;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(rows * heads * hd), ref(rows * heads * hd);
    for (auto & v : x) v = dist(rng);
    int positions[2] = {0, 5};
    for (int m = 0; m < rows; ++m) {
        for (int h = 0; h < heads; ++h) {
        for (int k = 0; k < hd / 2; ++k) {
            float angle = positions[m] * powf(theta, -2.0f * k / hd);
            float c = cosf(angle), s = sinf(angle);
            size_t base = (size_t) m * heads * hd + h * hd;
            ref[base + k]              = x[base + k] * c - x[base + k + hd / 2] * s;
            ref[base + k + hd / 2]     = x[base + k] * s + x[base + k + hd / 2] * c;
            }
        }
    }
    float * dx, * dout;
    int * dpos;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&dpos, rows * 4);
    cudaMalloc(&dout, x.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions, rows * 4, cudaMemcpyHostToDevice);
    rope_neox(dx, dpos, rows, heads, hd, theta, dout);
    std::vector<float> ours(x.size());
    cudaMemcpy(ours.data(), dout, ours.size() * 4, cudaMemcpyDeviceToHost);
    float mx = 0;
    for (size_t i = 0; i < ours.size(); ++i) mx = fmaxf(mx, fabsf(ours[i] - ref[i]));
    report("rope", mx < 1e-4f, mx, 1e-4f);
    cudaFree(dx); cudaFree(dpos); cudaFree(dout);
}

void test_attention() {
    std::mt19937 rng(4);
    const int hd = 64, n_head = 4, n_kv = 2, M = 3, max_seq = 8;
    const int nq = n_head * hd, nk = n_kv * hd;
    const int kv_stride = 2 * n_kv * hd * max_seq;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(M * nq), k(M * nk), v(M * nk), q_norm(hd), k_norm(hd);
    for (auto & x : q) x = dist(rng);
    for (auto & x : k) x = dist(rng);
    for (auto & x : v) x = dist(rng);
    for (auto & x : q_norm) x = dist(rng);
    for (auto & x : k_norm) x = dist(rng);
    int positions[M] = {0, 1, 3};
    int seqs[M] = {0, 0, 0};

    // CPU reference: same math as our kernels.
    auto norm_rope = [&](const std::vector<float> & xv, int m, int base_off, const std::vector<float> & wn,
                         int h, int kvh, int pos, float * out) {
        const float * xh = xv.data() + (size_t) m * base_off + h * hd;
        double ss = 0;
        for (int d = 0; d < hd; ++d) ss += (double) xh[d] * xh[d];
        float rms = 1.0f / sqrtf((float) (ss / hd) + 1e-6f);
        for (int d = 0; d < hd; ++d) out[d] = xh[d] * rms * wn[d];
        for (int kk = 0; kk < hd / 2; ++kk) {
            float angle = pos * powf(1000000.0f, -2.0f * kk / hd);
            float c = cosf(angle), s = sinf(angle);
            float a = out[kk], b = out[kk + hd / 2];
            out[kk]          = a * c - b * s;
            out[kk + hd / 2] = a * s + b * c;
        }
    };

    // KV cache: [seq, 2*n_kv*hd*max_seq] bf16; K region then V region.
    std::vector<bf16> cache(1 * kv_stride);
    std::vector<float> ref(M * nq);
    for (int m = 0; m < M; ++m) {
        float qn[256], kn[256];
        for (int h = 0; h < n_head; ++h) {
            int kvh = h / (n_head / n_kv);
            norm_rope(q, m, nq, q_norm, h, kvh, positions[m], qn);
            norm_rope(k, m, nk, k_norm, kvh, kvh, positions[m], kn);
            // store K/V (recompute for each shared kvh - idempotent)
            size_t koff = kvh * max_seq * hd + positions[m] * hd;
            size_t voff = n_kv * max_seq * hd + kvh * max_seq * hd + positions[m] * hd;
            for (int d = 0; d < hd; ++d) cache[koff + d] = __float2bfloat16(kn[d]);
        }
    }
    // Store proper V cache: overwrite V region with v values (normed? V is raw).
    for (int m = 0; m < M; ++m) {
        for (int h = 0; h < n_head; ++h) {
            int kvh = h / (n_head / n_kv);
            size_t voff = n_kv * max_seq * hd + kvh * max_seq * hd + positions[m] * hd;
            for (int d = 0; d < hd; ++d) {
                cache[voff + d] = __float2bfloat16(v[(size_t) m * nk + kvh * hd + d]);
            }
        }
    }
    // Attention
    int dbg_m = 1, dbg_h = 0, dbg_kvh = 0;
    printf("[selftest] attention ref debug m=%d h=%d kvh=%d:\n", dbg_m, dbg_h, dbg_kvh);
    for (int m = 0; m < M; ++m) {
        for (int h = 0; h < n_head; ++h) {
            int kvh = h / (n_head / n_kv);
            float qn[256], kn[256];
            norm_rope(q, m, nq, q_norm, h, kvh, positions[m], qn);
            norm_rope(k, m, nk, k_norm, kvh, kvh, positions[m], kn);
            if (m == dbg_m && h == dbg_h) {
                printf("  qn[0..3]=%g %g %g %g  kn[0..3]=%g %g %g %g\n",
                       qn[0], qn[1], qn[2], qn[3], kn[0], kn[1], kn[2], kn[3]);
                for (int p = 0; p <= positions[m]; ++p) {
                    printf("  cache K pos %d: %g %g %g %g\n", p,
                           __bfloat162float(cache[kvh * max_seq * hd + p * hd + 0]),
                           __bfloat162float(cache[kvh * max_seq * hd + p * hd + 1]),
                           __bfloat162float(cache[kvh * max_seq * hd + p * hd + 2]),
                           __bfloat162float(cache[kvh * max_seq * hd + p * hd + 3]));
                }
            }
            float scores[8];
            float smax = -1e30f;
            for (int p = 0; p <= positions[m]; ++p) {
                float s = 0;
                for (int d = 0; d < hd; ++d) {
                    bf16 kb = cache[kvh * max_seq * hd + p * hd + d];
                    s += qn[d] * __bfloat162float(kb);
                }
                scores[p] = s / sqrtf(hd);
                smax = fmaxf(smax, scores[p]);
            }
            float ssum = 0;
            for (int p = 0; p <= positions[m]; ++p) {
                scores[p] = expf(scores[p] - smax);
                ssum += scores[p];
            }
            for (int d = 0; d < hd; ++d) {
                float acc = 0;
                for (int p = 0; p <= positions[m]; ++p) {
                    bf16 vb = cache[n_kv * max_seq * hd + kvh * max_seq * hd + p * hd + d];
                    acc += scores[p] * __bfloat162float(vb);
                }
                ref[(size_t) m * nq + h * hd + d] = acc / ssum;
            }
        }
    }

    float * dq, * dk, * dv, * dqn, * dkn, * dout;
    int * dpos, * dseq;
    bf16 * dkv;
    cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dk, k.size() * 4); cudaMalloc(&dv, v.size() * 4);
    cudaMalloc(&dqn, q_norm.size() * 4); cudaMalloc(&dkn, k_norm.size() * 4);
    cudaMalloc(&dpos, M * 4); cudaMalloc(&dseq, M * 4); cudaMalloc(&dout, ref.size() * 4);
    cudaMalloc(&dkv, cache.size() * 2);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dqn, q_norm.data(), q_norm.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkn, k_norm.data(), k_norm.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions, M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dseq, seqs, M * 4, cudaMemcpyHostToDevice);
    cudaMemset(dkv, 0, cache.size() * 2);

    qk_norm_store(dq, dk, dv, nq, nk, nk, dqn, dkn, 1e-6f, 1000000.0f, hd, n_head, n_kv,
                  dpos, dseq, M, dkv, kv_stride, max_seq, dout);
    attention_compute(dout, dpos, dseq, M, dkv, kv_stride, max_seq, hd, n_head, n_kv, dout);
    std::vector<float> ours(ref.size());
    cudaMemcpy(ours.data(), dout, ours.size() * 4, cudaMemcpyDeviceToHost);

    float mx = 0;
    size_t argmx = 0;
    for (size_t i = 0; i < ours.size(); ++i) {
        float d = fabsf(ours[i] - ref[i]);
        if (d > mx) { mx = d; argmx = i; }
    }
    printf("[selftest] attention detail: maxdiff=%.6g at m=%zu h=%zu d=%zu ours=%.6g ref=%.6g\n",
           mx, argmx / nq, (argmx % nq) / hd, argmx % hd, ours[argmx], ref[argmx]);
    // full row for token 1, head 0
    size_t base = (size_t) 1 * nq;
    printf("[selftest] attention row m=1 h=0: ours=%g %g %g %g | ref=%g %g %g %g\n",
           ours[base], ours[base + 1], ours[base + 2], ours[base + 3],
           ref[base], ref[base + 1], ref[base + 2], ref[base + 3]);
    mx = 0;
    for (size_t i = 0; i < ours.size(); ++i) mx = fmaxf(mx, fabsf(ours[i] - ref[i]));
    report("attention", mx < 1e-3f, mx, 1e-3f);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dqn); cudaFree(dkn);
    cudaFree(dpos); cudaFree(dseq); cudaFree(dout); cudaFree(dkv);
}

void test_attention_real_dims() {
    // Same shapes as the real Qwen3-0.6B forward with multiple positions.
    std::mt19937 rng(7);
    const int hd = 128, n_head = 16, n_kv = 8, M = 2, max_seq = 16;
    const int nq = n_head * hd, nk = n_kv * hd;
    const int kv_stride = 2 * n_kv * hd * max_seq;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(M * nq), k(M * nk), v(M * nk), q_norm(hd), k_norm(hd);
    for (auto & x : q) x = dist(rng);
    for (auto & x : k) x = dist(rng);
    for (auto & x : v) x = dist(rng);
    for (auto & x : q_norm) x = dist(rng);
    for (auto & x : k_norm) x = dist(rng);
    int positions[M] = {0, 1};
    int seqs[M] = {0, 0};

    auto norm_rope = [&](const std::vector<float> & xv, int m, int base_off,
                         const std::vector<float> & wn, int h, int pos, float * out) {
        const float * xh = xv.data() + (size_t) m * base_off + h * hd;
        double ss = 0;
        for (int d = 0; d < hd; ++d) ss += (double) xh[d] * xh[d];
        float rms = 1.0f / sqrtf((float) (ss / hd) + 1e-6f);
        for (int d = 0; d < hd; ++d) out[d] = xh[d] * rms * wn[d];
        for (int kk = 0; kk < hd / 2; ++kk) {
            float angle = pos * powf(1000000.0f, -2.0f * kk / hd);
            float c = cosf(angle), s = sinf(angle);
            float a = out[kk], b = out[kk + hd / 2];
            out[kk]          = a * c - b * s;
            out[kk + hd / 2] = a * s + b * c;
        }
    };

    std::vector<bf16> cache(1 * kv_stride);
    for (int m = 0; m < M; ++m) {
        for (int h = 0; h < n_head; ++h) {
            int kvh = h / (n_head / n_kv);
            float kn[256];
            norm_rope(k, m, nk, k_norm, kvh, positions[m], kn);
            size_t koff = kvh * max_seq * hd + positions[m] * hd;
            for (int d = 0; d < hd; ++d) cache[koff + d] = __float2bfloat16(kn[d]);
            size_t voff = n_kv * max_seq * hd + kvh * max_seq * hd + positions[m] * hd;
            for (int d = 0; d < hd; ++d)
                cache[voff + d] = __float2bfloat16(v[(size_t) m * nk + kvh * hd + d]);
        }
    }
    std::vector<float> ref(M * nq);
    for (int m = 0; m < M; ++m) {
        for (int h = 0; h < n_head; ++h) {
            int kvh = h / (n_head / n_kv);
            float qn[256];
            norm_rope(q, m, nq, q_norm, h, positions[m], qn);
            float scores[16];
            float smax = -1e30f;
            for (int p = 0; p <= positions[m]; ++p) {
                float s = 0;
                for (int d = 0; d < hd; ++d)
                    s += qn[d] * __bfloat162float(cache[kvh * max_seq * hd + p * hd + d]);
                scores[p] = s / sqrtf(hd);
                smax = fmaxf(smax, scores[p]);
            }
            float ssum = 0;
            for (int p = 0; p <= positions[m]; ++p) {
                scores[p] = expf(scores[p] - smax);
                ssum += scores[p];
            }
            for (int d = 0; d < hd; ++d) {
                float acc = 0;
                for (int p = 0; p <= positions[m]; ++p)
                    acc += scores[p] * __bfloat162float(
                        cache[n_kv * max_seq * hd + kvh * max_seq * hd + p * hd + d]);
                ref[(size_t) m * nq + h * hd + d] = acc / ssum;
            }
        }
    }

    float * dq, * dk, * dv, * dqn, * dkn, * dout;
    int * dpos, * dseq;
    bf16 * dkv;
    cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dk, k.size() * 4); cudaMalloc(&dv, v.size() * 4);
    cudaMalloc(&dqn, q_norm.size() * 4); cudaMalloc(&dkn, k_norm.size() * 4);
    cudaMalloc(&dpos, M * 4); cudaMalloc(&dseq, M * 4); cudaMalloc(&dout, ref.size() * 4);
    cudaMalloc(&dkv, cache.size() * 2);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dqn, q_norm.data(), q_norm.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkn, k_norm.data(), k_norm.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions, M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dseq, seqs, M * 4, cudaMemcpyHostToDevice);
    cudaMemset(dkv, 0, cache.size() * 2);

    qk_norm_store(dq, dk, dv, nq, nk, nk, dqn, dkn, 1e-6f, 1000000.0f, hd, n_head, n_kv,
                  dpos, dseq, M, dkv, kv_stride, max_seq, dout);
    attention_compute(dout, dpos, dseq, M, dkv, kv_stride, max_seq, hd, n_head, n_kv, dout);
    std::vector<float> ours(ref.size());
    cudaMemcpy(ours.data(), dout, ours.size() * 4, cudaMemcpyDeviceToHost);

    float mx = 0;
    size_t argmx = 0;
    for (size_t i = 0; i < ours.size(); ++i) {
        float d = fabsf(ours[i] - ref[i]);
        if (d > mx) { mx = d; argmx = i; }
    }
    printf("[selftest] attention_real: maxdiff=%.6g at m=%zu h=%zu d=%zu ours=%.6g ref=%.6g\n",
           mx, argmx / nq, (argmx % nq) / hd, argmx % hd, ours[argmx], ref[argmx]);
    report("attention_real", mx < 1e-3f, mx, 1e-3f);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dqn); cudaFree(dkn);
    cudaFree(dpos); cudaFree(dseq); cudaFree(dout); cudaFree(dkv);
}

void bench_gemm() {
    std::mt19937 rng(9);
    auto A = random_bf16(1 * 1024, rng);
    auto B = random_bf16(151936 * 1024, rng);
    bf16 * da, * db;
    float * dc;
    cudaMalloc(&da, A.size() * 2);
    cudaMalloc(&db, B.size() * 2);
    cudaMalloc(&dc, (size_t) 1 * 151936 * 4);
    cudaMemcpy(da, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), B.size() * 2, cudaMemcpyHostToDevice);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    // warmup
    for (int i = 0; i < 10; ++i) gemm_bf16(da, db, dc, 1, 2048, 1024);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < 100; ++i) gemm_bf16(da, db, dc, 1, 2048, 1024);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[selftest] gemm bench M=1 N=2048 K=1024: %.3f ms/op (%.1f GFLOP/s)\n",
           ms / 100, 2.0 * 1 * 2048 * 1024 / (ms / 100 / 1000) / 1e9);

    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) gemm_bf16(da, db, dc, 1, 151936, 1024);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[selftest] gemm bench M=1 N=151936 K=1024: %.3f ms/op (%.1f GFLOP/s)\n",
           ms / 20, 2.0 * 1 * 151936 * 1024 / (ms / 20 / 1000) / 1e9);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(da); cudaFree(db); cudaFree(dc);
}

void test_gemm_real_shapes() {
    std::mt19937 rng(11);
    const int M = 2, K = 1024;
    for (auto [N, K] : std::vector<std::pair<int,int>>{{4096, 1024}, {2048, 1024}, {1024, 1024}, {1024, 3072}, {2048, 3072}}) {
        auto A = random_bf16(M * K, rng);
        auto B = random_bf16(N * K, rng);
        bf16 * da, * db;
        float * dc;
        cudaMalloc(&da, A.size() * 2);
        cudaMalloc(&db, B.size() * 2);
        cudaMalloc(&dc, (size_t) M * N * 4);
        cudaMemcpy(da, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(db, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
        gemm_bf16(da, db, dc, M, N, K);
        std::vector<float> ours(M * N);
        cudaMemcpy(ours.data(), dc, ours.size() * 4, cudaMemcpyDeviceToHost);
        float mx = 0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double s = 0;
                for (int k = 0; k < K; ++k)
                    s += __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
                mx = fmaxf(mx, fabsf(ours[m * N + n] - (float) s));
            }
        }
        printf("[selftest] gemm real M=%d N=%d K=%d: maxdiff=%.6g %s\n", M, N, K, mx,
               mx < 0.02f ? "PASS" : "FAIL");
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }
}

} // namespace

int run_selftest() {
    test_gemm();
    test_rms_norm();
    test_rope();
    test_attention();
    test_attention_real_dims();
    test_gemm_real_shapes();
    bench_gemm();
    printf("[selftest] %s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "FAILED", g_fail);
    return g_fail;
}
