#include "gemm.cuh"

namespace {

constexpr int TM = 64;
constexpr int TN = 64;
constexpr int TK = 32;
constexpr int THREADS = 128;

// Pack two bf16 values into one 32-bit register (little-endian: first = low).
__device__ __forceinline__ uint32_t pack2(bf16 a, bf16 b) {
    return (uint32_t) __bfloat16_as_ushort(a) |
           ((uint32_t) __bfloat16_as_ushort(b) << 16);
}

// ---------------------------------------------------------------------------
// Small-M GEMM (matrix-vector style). Used for decode (M <= 32) where the
// tiled mma kernel would waste most of its tile rows.
// C[M,N] = A[M,K] * B[N,K]^T ; B is the weight in W^T layout [N,K].
// ---------------------------------------------------------------------------
constexpr int GV_BN = 128;   // output columns per block
constexpr int GV_BK = 64;    // k-chunk
constexpr int GV_THREADS = 128;
constexpr int GV_BK_PAD = GV_BK + 2;   // pad to avoid bank conflicts
constexpr int GV_SPLIT = 4;  // split-K factor for small N (more parallelism)

__global__ void __launch_bounds__(GV_THREADS)
gemv_bf16_kernel(const bf16 * __restrict__ A, const bf16 * __restrict__ B,
                 float * __restrict__ C, int M, int N, int K, int ksplit) {
    __shared__ __align__(16) bf16 As[GV_BK];
    __shared__ __align__(16) bf16 Bs[GV_BN][GV_BK_PAD];

    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n0 = blockIdx.x * GV_BN;
    const int split = blockIdx.z;
    const int k_per_split = K / ksplit;
    const int k_begin = split * k_per_split;
    const int k_end = (split + 1) * k_per_split;
    const bf16 * arow = A + (size_t) m * K + k_begin;

    float acc = 0.0f;
    for (int k0 = k_begin; k0 < k_end; k0 += GV_BK) {
        // Load B tile [GV_BN][GV_BK] with 16-byte vector loads.
        for (int i = tid; i < (GV_BN * GV_BK) / 8; i += GV_THREADS) {
            const int idx = i * 8;
            const int r = idx / GV_BK;
            const int c = idx % GV_BK;
            const int gr = n0 + r;
            const int gc = k0 + c;
            uint4 v = (gr < N && gc + 7 < k_end)
                          ? *reinterpret_cast<const uint4 *>(B + (size_t) gr * K + gc)
                          : make_uint4(0, 0, 0, 0);
            uint32_t * dst = reinterpret_cast<uint32_t *>(&Bs[r][c]);
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }
        if (tid < GV_BK && k0 + tid < k_end) As[tid] = arow[k0 - k_begin + tid];
        __syncthreads();

        if (tid < GV_BN && n0 + tid < N) {
            const uint32_t * a32 = reinterpret_cast<const uint32_t *>(As);
            const uint32_t * b32 = reinterpret_cast<const uint32_t *>(Bs[tid]);
            for (int j = 0; j < GV_BK / 2; ++j) {
                uint32_t av = a32[j];
                uint32_t bv = b32[j];
                acc += __bfloat162float(*reinterpret_cast<const bf16 *>(&av)) *
                           __bfloat162float(*reinterpret_cast<const bf16 *>(&bv)) +
                       __bfloat162float(*(reinterpret_cast<const bf16 *>(&av) + 1)) *
                           __bfloat162float(*(reinterpret_cast<const bf16 *>(&bv) + 1));
            }
        }
        __syncthreads();
    }
    if (tid < GV_BN && n0 + tid < N) atomicAdd(&C[(size_t) m * N + n0 + tid], acc);
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
                 float * __restrict__ C, int M, int N, int K) {
    constexpr int WARP_ROWS = TM / 2;   // 32
    constexpr int WARP_COLS = TN / 2;   // 32

    __shared__ __align__(16) bf16 As[TM][TK];
    __shared__ __align__(16) bf16 Bs[TK][TN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_row = warp / 2;      // 0..1
    const int warp_col = warp % 2;      // 0..1

    const int m0 = blockIdx.x * TM;
    const int n0 = blockIdx.y * TN;

    float acc[2][4][4] = {};

    for (int k0 = 0; k0 < K; k0 += TK) {
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
            for (int ni = 0; ni < 4; ++ni) {
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

    // Epilogue
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int mbase = m0 + warp_row * WARP_ROWS + mi * 16;
            const int nbase = n0 + warp_col * WARP_COLS + ni * 8;
            const int r = lane / 4;
            const int c = (lane % 4) * 2;
            const float * accv = acc[mi][ni];
            if (mbase + r < M && nbase + c < N)
                C[(size_t) (mbase + r) * N + nbase + c] = accv[0];
            if (mbase + r < M && nbase + c + 1 < N)
                C[(size_t) (mbase + r) * N + nbase + c + 1] = accv[1];
            if (mbase + r + 8 < M && nbase + c < N)
                C[(size_t) (mbase + r + 8) * N + nbase + c] = accv[2];
            if (mbase + r + 8 < M && nbase + c + 1 < N)
                C[(size_t) (mbase + r + 8) * N + nbase + c + 1] = accv[3];
        }
    }
}

} // namespace

void gemm_bf16(const bf16 * A, const bf16 * B, float * C, int M, int N, int K,
               cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M <= 32) {
        QWEN_CU_CHECK(cudaMemsetAsync(C, 0, (size_t) M * N * sizeof(float), stream));
        const int ksplit = (N < 8192) ? GV_SPLIT : 1;
        dim3 grid((N + GV_BN - 1) / GV_BN, M, ksplit);
        gemv_bf16_kernel<<<grid, GV_THREADS, 0, stream>>>(A, B, C, M, N, K, ksplit);
        return;
    }
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    gemm_bf16_kernel<<<grid, THREADS, 0, stream>>>(A, B, C, M, N, K);
}
