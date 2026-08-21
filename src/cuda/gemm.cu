#include "gemm.cuh"

namespace {

constexpr int TM = 64;
constexpr int TN = 64;
constexpr int TK = 32;
constexpr int THREADS = 256;

// Pack two bf16 values into one 32-bit register (little-endian: first = low).
__device__ __forceinline__ uint32_t pack2(bf16 a, bf16 b) {
    return (uint32_t) __bfloat16_as_ushort(a) |
           ((uint32_t) __bfloat16_as_ushort(b) << 16);
}

// ---------------------------------------------------------------------------
// Small-M GEMM (matrix-vector style). Used for decode (M <= 32) where the
// tiled mma kernel would waste most of its tile rows.
// C[M,N] = A[M,K] * B[N,K]^T ; B is the weight in W^T layout [N,K].
//
// A100-oriented design: 256 threads per block (32 output columns x 8-way
// thread-level split-K). Every thread owns one (column, K-slice) and streams
// its K slice directly from global memory with 16-byte vector loads - no
// shared-memory tiles, no per-iteration __syncthreads. The 8 partial sums per
// column are reduced inside the block in fixed order, so results are
// bit-identical between eager launches and CUDA Graph replay. Engine shapes
// (K in {1024, 2048, 3072}, N a multiple of 32) take this fast path with zero
// bounds checks; other shapes fall back to the tiled mma kernel.
// ---------------------------------------------------------------------------
constexpr int GV2_BN = 32;       // output columns per block
constexpr int GV2_THREADS = 256; // = GV2_BN * GV2_KSPLIT
constexpr int GV2_KSPLIT = 8;    // thread-level split-K

constexpr int WGV_WARPS = 8;
constexpr int WGV_THREADS = WGV_WARPS * 32;

__global__ void __launch_bounds__(GV2_THREADS)
gemv2_bf16_kernel(const bf16 * __restrict__ A, const bf16 * __restrict__ B,
                  float * __restrict__ C, int M, int N, int K) {
    __shared__ float pbuf[GV2_KSPLIT][GV2_BN];

    const int m = blockIdx.y;
    const int n0 = blockIdx.x * GV2_BN;
    const int col = threadIdx.x & (GV2_BN - 1);   // GV2_BN = 32 (power of two)
    const int kp = threadIdx.x >> 5;              // 0..GV2_KSPLIT-1
    const int K_PER = K / GV2_KSPLIT;
    const int k_begin = kp * K_PER;

    // Both pointers are 16-byte aligned: K is a multiple of 64, k_begin is a
    // multiple of K_PER = K/8 (so a multiple of 8 bf16 = 16 bytes).
    const uint4 * a4 = reinterpret_cast<const uint4 *>(A + (size_t) m * K + k_begin);
    const uint4 * b4 = reinterpret_cast<const uint4 *>(B + (size_t)(n0 + col) * K + k_begin);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int iters = K_PER / 8;                  // 8 bf16 per uint4
    for (int i = 0; i < iters; i += 4) {
        const uint4 av0 = a4[i], av1 = a4[i + 1], av2 = a4[i + 2], av3 = a4[i + 3];
        const uint4 bv0 = b4[i], bv1 = b4[i + 1], bv2 = b4[i + 2], bv3 = b4[i + 3];
        const bf16 * ap0 = reinterpret_cast<const bf16 *>(&av0);
        const bf16 * bp0 = reinterpret_cast<const bf16 *>(&bv0);
        const bf16 * ap1 = reinterpret_cast<const bf16 *>(&av1);
        const bf16 * bp1 = reinterpret_cast<const bf16 *>(&bv1);
        const bf16 * ap2 = reinterpret_cast<const bf16 *>(&av2);
        const bf16 * bp2 = reinterpret_cast<const bf16 *>(&bv2);
        const bf16 * ap3 = reinterpret_cast<const bf16 *>(&av3);
        const bf16 * bp3 = reinterpret_cast<const bf16 *>(&bv3);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc0 += bf16_to_f32(ap0[j]) * bf16_to_f32(bp0[j]);
            acc1 += bf16_to_f32(ap1[j]) * bf16_to_f32(bp1[j]);
            acc2 += bf16_to_f32(ap2[j]) * bf16_to_f32(bp2[j]);
            acc3 += bf16_to_f32(ap3[j]) * bf16_to_f32(bp3[j]);
        }
    }
    // Deterministic block-level reduction over the GV2_KSPLIT partial sums.
    const float acc = (acc0 + acc1) + (acc2 + acc3);
    pbuf[kp][col] = acc;
    __syncthreads();
    if (kp == 0) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < GV2_KSPLIT; ++k) s += pbuf[k][col];
        C[(size_t) m * N + n0 + col] = s;
    }
}

// SM120 decode path. Each warp owns one output row, so adjacent lanes read
// adjacent BF16 weights. This is substantially more bandwidth-efficient on
// Blackwell than assigning adjacent lanes to K-strided output rows.
__global__ void __launch_bounds__(WGV_THREADS)
gemv_warp_bf16_kernel(const bf16 * __restrict__ A, const bf16 * __restrict__ B,
                      float * __restrict__ C, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int m = blockIdx.y;
    const int n = blockIdx.x * WGV_WARPS + warp;
    if (m >= M || n >= N) return;

    const bf16 * a = A + (size_t) m * K;
    const bf16 * b = B + (size_t) n * K;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int k = lane * 8; k < K; k += 32 * 8) {
        const uint4 av = *reinterpret_cast<const uint4 *>(a + k);
        const uint4 bv = *reinterpret_cast<const uint4 *>(b + k);
        const bf16 * ap = reinterpret_cast<const bf16 *>(&av);
        const bf16 * bp = reinterpret_cast<const bf16 *>(&bv);
        acc0 = fmaf(bf16_to_f32(ap[0]), bf16_to_f32(bp[0]), acc0);
        acc1 = fmaf(bf16_to_f32(ap[1]), bf16_to_f32(bp[1]), acc1);
        acc2 = fmaf(bf16_to_f32(ap[2]), bf16_to_f32(bp[2]), acc2);
        acc3 = fmaf(bf16_to_f32(ap[3]), bf16_to_f32(bp[3]), acc3);
        acc0 = fmaf(bf16_to_f32(ap[4]), bf16_to_f32(bp[4]), acc0);
        acc1 = fmaf(bf16_to_f32(ap[5]), bf16_to_f32(bp[5]), acc1);
        acc2 = fmaf(bf16_to_f32(ap[6]), bf16_to_f32(bp[6]), acc2);
        acc3 = fmaf(bf16_to_f32(ap[7]), bf16_to_f32(bp[7]), acc3);
    }

    float sum = (acc0 + acc1) + (acc2 + acc3);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) C[(size_t) m * N + n] = sum;
}

bool use_sm120_warp_gemv() {
    static const bool enabled = [] {
        if (getenv("QWEN_GEMV_LEGACY")) return false;
        int device = 0;
        cudaDeviceProp properties{};
        QWEN_CU_CHECK(cudaGetDevice(&device));
        QWEN_CU_CHECK(cudaGetDeviceProperties(&properties, device));
        return properties.major == 12;
    }();
    return enabled;
}

// mma.m16n8k16 row-major A (16x16), col-major B (16x8), FP32 accum.
__device__ __forceinline__ void mma_m16n8k16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1,
                                             float & c0, float & c1, float & c2, float & c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(THREADS)
gemm_bf16_kernel(const bf16 * __restrict__ A, const bf16 * __restrict__ B,
                 float * __restrict__ C_partial, int M, int N, int K, int SK) {
    constexpr int WARP_ROWS = TM / 2;   // 32 (2 row-warps)
    constexpr int WARP_COLS = TN / 2;   // 16 (4 col-warps x 2 m16n8 = 64 cols)

    __shared__ __align__(16) bf16 As[TM][TK];
    __shared__ __align__(16) bf16 Bs[TK][TN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_row = warp / 4;      // 0..1 (8 warps, 2x4 layout)
    const int warp_col = warp % 4;      // 0..3

    const int m0 = blockIdx.x * TM;
    const int n0 = blockIdx.y * TN;
    const int sk = blockIdx.z;
    // This block's K slice: split-K adds more concurrent blocks (and hence
    // more in-flight tensor-core work) for the small-M prefill GEMMs.
    const int K_PER = (K + SK - 1) / SK;
    const int k_begin = sk * K_PER;
    const int k_end = min(k_begin + K_PER, K);

    float acc[2][2][4] = {};

    for (int k0 = k_begin; k0 < k_end; k0 += TK) {
        // Cooperative load A tile [TM, TK] and B tile [TK, TN] into shared.
        for (int i = tid; i < TM * TK; i += THREADS) {
            int r = i / TK;
            int c = i % TK;
            int gr = m0 + r;
            int gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[(size_t) gr * K + gc] : f32_to_bf16(0.0f);
        }
        for (int i = tid; i < TK * TN; i += THREADS) {
            int r = i / TN;              // k index
            int c = i % TN;              // n index
            int gr = n0 + c;
            int gc = k0 + r;
            Bs[r][c] = (gr < N && gc < K) ? B[(size_t) gr * K + gc] : f32_to_bf16(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int mbase = warp_row * WARP_ROWS + mi * 16;
                const int nbase = warp_col * WARP_COLS + ni * 8;

                // TK = 32 covers two mma.m16n8k16 k-steps.
#pragma unroll
                for (int ks = 0; ks < TK; ks += 16) {
                    // A fragment: 16x16 row-major (lane/4 rows, (lane%4)*2 cols)
                    const int arow = lane / 4;
                    const int acol = (lane % 4) * 2 + ks;
                    uint32_t a0 = pack2(As[mbase + arow][acol], As[mbase + arow][acol + 1]);
                    uint32_t a1 = pack2(As[mbase + arow + 8][acol], As[mbase + arow + 8][acol + 1]);
                    uint32_t a2 = pack2(As[mbase + arow][acol + 8], As[mbase + arow][acol + 9]);
                    uint32_t a3 = pack2(As[mbase + arow + 8][acol + 8], As[mbase + arow + 8][acol + 9]);

                    // B fragment: 16x8 col-major, i.e. rows k = (lane%4)*2, col n = lane/4
                    const int brow = (lane % 4) * 2 + ks;
                    const int bcol = lane / 4;
                    uint32_t b0 = pack2(Bs[brow][nbase + bcol], Bs[brow + 1][nbase + bcol]);
                    uint32_t b1 = pack2(Bs[brow + 8][nbase + bcol], Bs[brow + 9][nbase + bcol]);

                    float * c = acc[mi][ni];
                    mma_m16n8k16(a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: write this split's partial result (SK=1 writes C directly via
    // the caller passing C as C_partial).
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int mbase = m0 + warp_row * WARP_ROWS + mi * 16;
            const int nbase = n0 + warp_col * WARP_COLS + ni * 8;
            const int r = lane / 4;
            const int c = (lane % 4) * 2;
            const float * accv = acc[mi][ni];
            const size_t rowbase = (size_t) (mbase + r) * N + nbase + c;
            if (mbase + r < M && nbase + c < N)
                C_partial[rowbase * SK + sk] = accv[0];
            if (mbase + r < M && nbase + c + 1 < N)
                C_partial[(rowbase + 1) * SK + sk] = accv[1];
            if (mbase + r + 8 < M && nbase + c < N)
                C_partial[((size_t) (mbase + r + 8) * N + nbase + c) * SK + sk] = accv[2];
            if (mbase + r + 8 < M && nbase + c + 1 < N)
                C_partial[((size_t) (mbase + r + 8) * N + nbase + c + 1) * SK + sk] = accv[3];
        }
    }
}

// Deterministic split-K reduction: C[m,n] = sum of partials in fixed sk order.
__global__ void gemm_bf16_reduce_kernel(const float * __restrict__ C_partial,
                                        float * __restrict__ C, int M, int N, int SK) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const float * p = C_partial + (size_t) idx * SK;
    float s = 0.0f;
    for (int k = 0; k < SK; ++k) s += p[k];
    C[idx] = s;
}

} // namespace

void gemm_bf16(const bf16 * A, const bf16 * B, float * C, int M, int N, int K,
               cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M <= 32) {
        // Fast path: engine shapes (K multiple of 64 -> 16B-aligned vector
        // loads, N multiple of 32) use a shared-memory-free GEMV kernel.
        // Anything else falls back to the bounds-checked tiled MMA kernel.
        const bool fast = (N % GV2_BN == 0) && (K % 64 == 0);
        if (fast) {
            if (use_sm120_warp_gemv()) {
                dim3 grid((N + WGV_WARPS - 1) / WGV_WARPS, M);
                gemv_warp_bf16_kernel<<<grid, WGV_THREADS, 0, stream>>>(A, B, C, M, N, K);
                return;
            }
            dim3 grid(N / GV2_BN, M);
            gemv2_bf16_kernel<<<grid, GV2_THREADS, 0, stream>>>(A, B, C, M, N, K);
            return;
        }
        gemm_bf16_kernel<<<dim3((M + TM - 1) / TM, (N + TN - 1) / TN), THREADS, 0, stream>>>(
            A, B, C, M, N, K, 1);
        return;
    }
    // M > 32: tiled mma kernel. Split-K (2-way) doubles the number of blocks
    // for the small-M prefill GEMMs, giving the tensor cores more concurrent
    // work; the partials are reduced deterministically (fixed split order).
    const int SK = (K >= 1024 && N >= 256) ? 2 : 1;
    if (SK > 1) {
        static thread_local float * partial = nullptr;
        static thread_local size_t partial_bytes = 0;
        const size_t need = (size_t) M * N * SK * sizeof(float);
        if (partial_bytes < need) {
            if (partial) cudaFree(partial);
            QWEN_CU_CHECK(cudaMalloc(&partial, need));
            partial_bytes = need;
        }
        dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN, SK);
        gemm_bf16_kernel<<<grid, THREADS, 0, stream>>>(A, B, partial, M, N, K, SK);
        const int total = M * N;
        gemm_bf16_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(partial, C, M, N, SK);
        return;
    }
    gemm_bf16_kernel<<<dim3((M + TM - 1) / TM, (N + TN - 1) / TN), THREADS, 0, stream>>>(
        A, B, C, M, N, K, 1);
}
