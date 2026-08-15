#pragma once
// Common CUDA helpers for qwen3-gpu-infer.
// Self-contained: no ggml/llama.cpp includes. BF16 vector helpers, error checks.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

using bf16 = __nv_bfloat16;

#define QWEN_CU_CHECK(call)                                                          \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(err_));                                       \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define QWEN_CU_CHECK_LAST() QWEN_CU_CHECK(cudaGetLastError())

#if defined(__CUDACC__)

// 128-bit vector of 8 bf16 values (2 uint4s) - used for wide loads/stores.
struct bf16x8 {
    uint32_t v[4];
};

static inline __device__ float bf16_to_f32(bf16 v) {
    return __bfloat162float(v);
}

static inline __device__ bf16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

static inline __device__ bf16x8 load_bf16x8(const bf16 * p) {
    bf16x8 r;
    const uint32_t * q = reinterpret_cast<const uint32_t *>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r.v[i] = __ldg(q + i);
    }
    return r;
}

static inline __device__ void store_bf16x8(bf16 * p, const bf16x8 & r) {
    uint32_t * q = reinterpret_cast<uint32_t *>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = r.v[i];
    }
}

// float4 vector loads for FP32 activations.
static inline __device__ float4 load_f32x4(const float * p) {
    return *reinterpret_cast<const float4 *>(p);
}

static inline __device__ void store_f32x4(float * p, const float4 & v) {
    *reinterpret_cast<float4 *>(p) = v;
}

// Block-wide reductions (works for any blockDim multiple of 32, <= 1024).
template <int ID>
static inline __device__ float block_reduce_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

template <int ID>
static inline __device__ float block_reduce_max(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : -INFINITY;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

#endif // __CUDACC__
