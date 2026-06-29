// SPDX-License-Identifier: Apache-2.0
// Structure adapted from vLLM's rms_norm_per_block_quant fused op; RTP-LLM
// exposes it as an eager QuantizedActivation producer for SM120 FP8 linear.
#include "rtp_llm/models_py/bindings/cuda/kernels/rms_norm_per_block_quant.h"

#if USING_CUDA
#include "vec_dtypes.cuh"
#include "util.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#endif

namespace rtp_llm {

#if USING_CUDA

__device__ __forceinline__ float rmsQuantWarpReduceSum(float val) {
    constexpr unsigned mask = 0xffffffff;
    val += __shfl_xor_sync(mask, val, 16);
    val += __shfl_xor_sync(mask, val, 8);
    val += __shfl_xor_sync(mask, val, 4);
    val += __shfl_xor_sync(mask, val, 2);
    val += __shfl_xor_sync(mask, val, 1);
    return val;
}

__device__ __forceinline__ float rmsQuantBlockReduceSumByWarp(float val, float* warp_sums) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps   = (blockDim.x + 31) >> 5;

    val = rmsQuantWarpReduceSum(val);
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    val = threadIdx.x < warps ? warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        val = rmsQuantWarpReduceSum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

__device__ __forceinline__ float rmsQuantBlockReduceSum(float val, float* smem) {
    smem[threadIdx.x] = val;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

template<int THREADS_PER_GROUP>
__device__ __forceinline__ float rmsQuantGroupReduceMax(float val) {
    static_assert(THREADS_PER_GROUP == 1 || THREADS_PER_GROUP == 2 || THREADS_PER_GROUP == 4
                  || THREADS_PER_GROUP == 8 || THREADS_PER_GROUP == 16 || THREADS_PER_GROUP == 32);
    const int warp_lane     = threadIdx.x & 31;
    const int subwarp_start = warp_lane & ~(THREADS_PER_GROUP - 1);
    unsigned  mask          = 0xffffffffu;
    if constexpr (THREADS_PER_GROUP < 32) {
        mask = ((1u << THREADS_PER_GROUP) - 1u) << subwarp_start;
    }
    if constexpr (THREADS_PER_GROUP >= 32) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, 16));
    }
    if constexpr (THREADS_PER_GROUP >= 16) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
    }
    if constexpr (THREADS_PER_GROUP >= 8) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
    }
    if constexpr (THREADS_PER_GROUP >= 4) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
    }
    if constexpr (THREADS_PER_GROUP >= 2) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
    }
    return val;
}

__device__ __forceinline__ void rmsQuantStGlobalInt4(int4* ptr, const int4& value) {
    asm volatile(
        "st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__ __forceinline__ float2 rmsQuantFmul2Rn(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    return __fmul2_rn(a, b);
#else
    float2 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    return result;
#endif
}

template<typename T>
__global__ void rmsNormPerBlockQuantFp8RuntimeGroupKernel(const T* __restrict__ input,
                                                          __nv_fp8_e4m3* __restrict__ output_q,
                                                          float* __restrict__ output_s,
                                                          const T* __restrict__ weight,
                                                          const int   hidden_size,
                                                          const int   hidden_groups,
                                                          const int   group_size,
                                                          const int   scale_token_stride,
                                                          const int   scale_hidden_stride,
                                                          const float rms_eps,
                                                          const float quant_eps,
                                                          const float max_8bit) {
    // Correctness fallback for uncommon group sizes or very wide rows. It keeps
    // the original RMSNorm -> quant arithmetic shape and chunks groups so
    // hidden_groups may exceed blockDim.x.
    extern __shared__ float smem[];
    float* reduce_smem = smem;

    const int token_idx = blockIdx.x;
    const T*  row_input = input + token_idx * hidden_size;
    auto output_pair    = reinterpret_cast<__nv_fp8x2_storage_t*>(output_q + token_idx * hidden_size);

    constexpr int VEC_SIZE = 4;
    const int     group_vec_size = group_size / VEC_SIZE;
    const int     vec_hidden_size = hidden_size / VEC_SIZE;

    float local_square_sum = 0.0f;
    for (int vec_idx = threadIdx.x; vec_idx < vec_hidden_size; vec_idx += blockDim.x) {
        vec_t<T, VEC_SIZE> input_vec;
        input_vec.load(row_input + vec_idx * VEC_SIZE);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            const float val = static_cast<float>(input_vec[i]);
            local_square_sum += val * val;
        }
    }
    const float square_sum = rmsQuantBlockReduceSum(local_square_sum, smem);
    const float rms        = rsqrtf(square_sum / hidden_size + rms_eps);

    int threads_per_group = 1;
    while ((threads_per_group << 1) <= group_vec_size && (threads_per_group << 1) * hidden_groups <= blockDim.x) {
        threads_per_group <<= 1;
    }
    const int groups_per_iter = static_cast<int>(blockDim.x) / threads_per_group;
    for (int group_base = 0; group_base < hidden_groups; group_base += groups_per_iter) {
        const int local_group_idx   = threadIdx.x / threads_per_group;
        const int thread_group      = threadIdx.x - local_group_idx * threads_per_group;
        const int group_idx         = group_base + local_group_idx;
        const int remaining_groups  = hidden_groups - group_base;
        const int active_group_size = remaining_groups < groups_per_iter ? remaining_groups : groups_per_iter;
        const int active_threads    = active_group_size * threads_per_group;

        float local_absmax = quant_eps;
        if (threadIdx.x < active_threads) {
            const int group_offset = group_idx * group_size;
            for (int vec_j = thread_group; vec_j < group_vec_size; vec_j += threads_per_group) {
                const int hidden_idx = group_offset + vec_j * VEC_SIZE;
                vec_t<T, VEC_SIZE> input_vec;
                vec_t<T, VEC_SIZE> weight_vec;
                input_vec.load(row_input + hidden_idx);
                weight_vec.load(weight + hidden_idx);
#pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    const float normalized = static_cast<float>(input_vec[i]) * rms;
                    const T     lowprec    = static_cast<T>(normalized * static_cast<float>(weight_vec[i]));
                    local_absmax           = fmaxf(local_absmax, fabsf(static_cast<float>(lowprec)));
                }
            }
        }
        reduce_smem[threadIdx.x] = local_absmax;
        __syncthreads();

        for (int stride = 1; stride < threads_per_group; stride <<= 1) {
            if (threadIdx.x < active_threads) {
                const int group_thread = threadIdx.x % threads_per_group;
                if ((group_thread % (stride << 1)) == 0 && group_thread + stride < threads_per_group) {
                    reduce_smem[threadIdx.x] = fmaxf(reduce_smem[threadIdx.x], reduce_smem[threadIdx.x + stride]);
                }
            }
            __syncthreads();
        }

        if (threadIdx.x < active_threads && threadIdx.x % threads_per_group == 0) {
            const int   output_group_idx = group_base + threadIdx.x / threads_per_group;
            const float absmax           = reduce_smem[threadIdx.x];
            reduce_smem[threadIdx.x]     = absmax / max_8bit;
            output_s[token_idx * scale_token_stride + output_group_idx * scale_hidden_stride] = absmax / max_8bit;
        }
        __syncthreads();

        if (threadIdx.x < active_threads) {
            const float scale        = reduce_smem[local_group_idx * threads_per_group];
            const int   group_offset = group_idx * group_size;
            for (int vec_j = thread_group; vec_j < group_vec_size; vec_j += threads_per_group) {
                const int hidden_idx = group_offset + vec_j * VEC_SIZE;
                vec_t<T, VEC_SIZE> input_vec;
                vec_t<T, VEC_SIZE> weight_vec;
                input_vec.load(row_input + hidden_idx);
                weight_vec.load(weight + hidden_idx);

                float output_values[VEC_SIZE];
#pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    const float normalized = static_cast<float>(input_vec[i]) * rms;
                    const T     lowprec    = static_cast<T>(normalized * static_cast<float>(weight_vec[i]));
                    output_values[i]       = fminf(fmaxf(static_cast<float>(lowprec) / scale, -max_8bit), max_8bit);
                }

                const float2 outputx2_0      = {output_values[0], output_values[1]};
                const float2 outputx2_1      = {output_values[2], output_values[3]};
                const int    vec_idx         = hidden_idx / VEC_SIZE;
                output_pair[vec_idx * 2]     = __nv_cvt_float2_to_fp8x2(outputx2_0, __NV_SATFINITE, __NV_E4M3);
                output_pair[vec_idx * 2 + 1] = __nv_cvt_float2_to_fp8x2(outputx2_1, __NV_SATFINITE, __NV_E4M3);
            }
        }
        __syncthreads();
    }
}

template<typename T, int GROUP_SIZE>
__global__ void rmsNormPerBlockQuantFp8VecKernel(const T* __restrict__ input,
                                                 __nv_fp8_e4m3* __restrict__ output_q,
                                                 float* __restrict__ output_s,
                                                 const T* __restrict__ weight,
                                                 const int   hidden_size,
                                                 const int   hidden_groups,
                                                 const int   scale_token_stride,
                                                 const int   scale_hidden_stride,
                                                 const float rms_eps,
                                                 const float quant_eps,
                                                 const float max_8bit) {
    // Production fast path: one 32B activation vector per active thread. This
    // avoids materializing the RMSNorm output and reuses the loaded activation
    // vector across RMS reduction and quantization.
    constexpr int THREADS_PER_GROUP = GROUP_SIZE / 16;
    constexpr int VEC_SIZE          = 16;
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32 || GROUP_SIZE == 64 || GROUP_SIZE == 128);
    const int     active_threads    = hidden_groups * THREADS_PER_GROUP;

    __shared__ float warp_sums[32];

    const int token_idx = blockIdx.x;
    const T*  row_input = input + token_idx * hidden_size;

    float local_square_sum = 0.0f;
    // Keep the loaded activation vector live across RMS reduction so the
    // quant pass does not read the same input row from HBM again.
    vec_t<T, VEC_SIZE> input_vec;
    if (threadIdx.x < active_threads) {
        const int group_idx  = threadIdx.x / THREADS_PER_GROUP;
        const int lane_id    = threadIdx.x - group_idx * THREADS_PER_GROUP;
        const int hidden_idx = group_idx * GROUP_SIZE + lane_id * VEC_SIZE;

        input_vec.load(row_input + hidden_idx);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            const float val = static_cast<float>(input_vec[i]);
            local_square_sum += val * val;
        }
    }
    const float square_sum = rmsQuantBlockReduceSumByWarp(local_square_sum, warp_sums);
    const float rms        = rsqrtf(square_sum / hidden_size + rms_eps);

    if (threadIdx.x >= active_threads) {
        return;
    }

    const int group_idx  = threadIdx.x / THREADS_PER_GROUP;
    const int lane_id    = threadIdx.x - group_idx * THREADS_PER_GROUP;
    const int hidden_idx = group_idx * GROUP_SIZE + lane_id * VEC_SIZE;

    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<T, VEC_SIZE> lowprec_vec;
    weight_vec.load(weight + hidden_idx);

    float local_absmax = quant_eps;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        const float normalized = static_cast<float>(input_vec[i]) * rms;
        const T     lowprec    = static_cast<T>(normalized * static_cast<float>(weight_vec[i]));
        lowprec_vec[i]         = lowprec;
        local_absmax           = fmaxf(local_absmax, fabsf(static_cast<float>(lowprec)));
    }

    const float absmax    = rmsQuantGroupReduceMax<THREADS_PER_GROUP>(local_absmax);
    const float inv_scale = max_8bit / absmax;
    if (lane_id == 0) {
        output_s[token_idx * scale_token_stride + group_idx * scale_hidden_stride] = absmax / max_8bit;
    }

    int4 output_buf;
    auto output_pair = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
    const float2 scale_repeated = {inv_scale, inv_scale};
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 2) {
        const float2 inputx2 = {static_cast<float>(lowprec_vec[i]), static_cast<float>(lowprec_vec[i + 1])};
        float2       outputx2 = rmsQuantFmul2Rn(inputx2, scale_repeated);
        outputx2.x      = fminf(fmaxf(outputx2.x, -max_8bit), max_8bit);
        outputx2.y      = fminf(fmaxf(outputx2.y, -max_8bit), max_8bit);
        output_pair[i / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
    }

    rmsQuantStGlobalInt4(reinterpret_cast<int4*>(output_q + token_idx * hidden_size + hidden_idx), output_buf);
}

template<typename T>
void launchRmsNormPerBlockQuantFp8RuntimeGroup(const torch::Tensor& input,
                                               const torch::Tensor& output_q,
                                               const torch::Tensor& output_s,
                                               const torch::Tensor& weight,
                                               const int   tokens,
                                               const int   hidden_size,
                                               const int   hidden_groups,
                                               const int   group_size,
                                               const int   scale_token_stride,
                                               const int   scale_group_stride,
                                               const float rms_eps,
                                               const float quant_eps,
                                               const float max_8bit,
                                               cudaStream_t stream) {
    const int block_size      = hidden_groups > 512 ? 1024 : (hidden_groups > 256 ? 512 : 256);
    const int shared_mem_size = block_size * sizeof(float);
    rmsNormPerBlockQuantFp8RuntimeGroupKernel<T><<<dim3(tokens), dim3(block_size), shared_mem_size, stream>>>(
        static_cast<T*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<T*>(weight.data_ptr()),
        hidden_size,
        hidden_groups,
        group_size,
        scale_token_stride,
        scale_group_stride,
        rms_eps,
        quant_eps,
        max_8bit);
}

template<typename T, int GROUP_SIZE>
void launchRmsNormPerBlockQuantFp8Vec(const torch::Tensor& input,
                                      const torch::Tensor& output_q,
                                      const torch::Tensor& output_s,
                                      const torch::Tensor& weight,
                                      const int   tokens,
                                      const int   hidden_size,
                                      const int   hidden_groups,
                                      const int   scale_token_stride,
                                      const int   scale_group_stride,
                                      const float rms_eps,
                                      const float quant_eps,
                                      const float max_8bit,
                                      cudaStream_t stream) {
    const int active_threads = hidden_groups * (GROUP_SIZE / 16);
    const int block_size     = active_threads > 512 ? 1024 : (active_threads > 256 ? 512 : 256);
    rmsNormPerBlockQuantFp8VecKernel<T, GROUP_SIZE><<<dim3(tokens), dim3(block_size), 0, stream>>>(
        static_cast<T*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<T*>(weight.data_ptr()),
        hidden_size,
        hidden_groups,
        scale_token_stride,
        scale_group_stride,
        rms_eps,
        quant_eps,
        max_8bit);
}

template<typename T>
void launchRmsNormPerBlockQuantFp8VecByGroupSize(const torch::Tensor& input,
                                                 const torch::Tensor& output_q,
                                                 const torch::Tensor& output_s,
                                                 const torch::Tensor& weight,
                                                 const int   tokens,
                                                 const int   hidden_size,
                                                 const int   hidden_groups,
                                                 const int   group_size,
                                                 const int   scale_token_stride,
                                                 const int   scale_group_stride,
                                                 const float rms_eps,
                                                 const float quant_eps,
                                                 const float max_8bit,
                                                 cudaStream_t stream) {
    switch (group_size) {
        case 16:
            if (hidden_groups <= 1024) {
                launchRmsNormPerBlockQuantFp8Vec<T, 16>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            } else {
                launchRmsNormPerBlockQuantFp8RuntimeGroup<T>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, 16, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            }
            return;
        case 32:
            if (hidden_groups * 2 <= 1024) {
                launchRmsNormPerBlockQuantFp8Vec<T, 32>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            } else {
                launchRmsNormPerBlockQuantFp8RuntimeGroup<T>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, 32, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            }
            return;
        case 64:
            if (hidden_groups * 4 <= 1024) {
                launchRmsNormPerBlockQuantFp8Vec<T, 64>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            } else {
                launchRmsNormPerBlockQuantFp8RuntimeGroup<T>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, 64, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            }
            return;
        case 128:
            if (hidden_groups * 8 <= 1024) {
                launchRmsNormPerBlockQuantFp8Vec<T, 128>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            } else {
                launchRmsNormPerBlockQuantFp8RuntimeGroup<T>(
                    input, output_q, output_s, weight, tokens, hidden_size, hidden_groups, 128, scale_token_stride,
                    scale_group_stride, rms_eps, quant_eps, max_8bit, stream);
            }
            return;
        default:
            TORCH_CHECK(false, "Unsupported group_size");
    }
}

template<typename T>
void launchRmsNormPerBlockQuantFp8ByGroupSize(const torch::Tensor& input,
                                              const torch::Tensor& output_q,
                                              const torch::Tensor& output_s,
                                              const torch::Tensor& weight,
                                              const int   tokens,
                                              const int   hidden_size,
                                              const int   hidden_groups,
                                              const int   group_size,
                                              const int   scale_token_stride,
                                              const int   scale_group_stride,
                                              const float rms_eps,
                                              const float quant_eps,
                                              const float max_8bit,
                                              cudaStream_t stream) {
    switch (group_size) {
        case 16:
        case 32:
        case 64:
        case 128:
            launchRmsNormPerBlockQuantFp8VecByGroupSize<T>(input,
                                                           output_q,
                                                           output_s,
                                                           weight,
                                                           tokens,
                                                           hidden_size,
                                                           hidden_groups,
                                                           group_size,
                                                           scale_token_stride,
                                                           scale_group_stride,
                                                           rms_eps,
                                                           quant_eps,
                                                           max_8bit,
                                                           stream);
            return;
        default:
            launchRmsNormPerBlockQuantFp8RuntimeGroup<T>(input,
                                                         output_q,
                                                         output_s,
                                                         weight,
                                                         tokens,
                                                         hidden_size,
                                                         hidden_groups,
                                                         group_size,
                                                         scale_token_stride,
                                                         scale_group_stride,
                                                         rms_eps,
                                                         quant_eps,
                                                         max_8bit,
                                                         stream);
            return;
    }
}

void rms_norm_per_block_quant_fp8(torch::Tensor input,
                                  torch::Tensor output_q,
                                  torch::Tensor output_s,
                                  torch::Tensor weight,
                                  int64_t       group_size,
                                  double        rms_eps,
                                  double        quant_eps,
                                  double        min_8bit,
                                  double        max_8bit) {
    CHECK_INPUT(input);
    CHECK_INPUT(output_q);
    CHECK_INPUT(weight);
    CHECK_CUDA(output_s);
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
    TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q shape must match input shape");
    TORCH_CHECK(output_s.dim() == 2, "output_s must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight size must match input hidden size");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input dtype");
    TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output_q must be float8_e4m3fn");
    TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Float, "output_s must be float32");
    TORCH_CHECK(min_8bit == -max_8bit, "only symmetric quantization is supported");
    TORCH_CHECK(rms_eps > 0.0, "rms_eps must be positive");
    TORCH_CHECK(quant_eps > 0.0, "quant_eps must be positive");
    TORCH_CHECK(input.size(1) % 4 == 0, "hidden size must be divisible by 4");
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0, "group_size must be positive and divisible by 4");
    const int tokens        = static_cast<int>(input.size(0));
    const int hidden_size   = static_cast<int>(input.size(1));
    const int hidden_groups = static_cast<int>(hidden_size / group_size);
    TORCH_CHECK(tokens > 0, "input must have at least one token");
    TORCH_CHECK(hidden_size > 0 && hidden_size % group_size == 0, "hidden size must be divisible by group_size");
    TORCH_CHECK(output_s.size(0) == tokens && output_s.size(1) == hidden_groups,
                "output_s shape must be (tokens, hidden_size / group_size)");
    TORCH_CHECK(output_s.numel() >= tokens * hidden_groups, "output_s buffer is too small");

    const int  scale_token_stride = static_cast<int>(output_s.stride(0));
    const int  scale_group_stride = static_cast<int>(output_s.stride(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
        launchRmsNormPerBlockQuantFp8ByGroupSize<scalar_t>(input,
                                                           output_q,
                                                           output_s,
                                                           weight,
                                                           tokens,
                                                           hidden_size,
                                                           hidden_groups,
                                                           static_cast<int>(group_size),
                                                           scale_token_stride,
                                                           scale_group_stride,
                                                           static_cast<float>(rms_eps),
                                                           static_cast<float>(quant_eps),
                                                           static_cast<float>(max_8bit),
                                                           stream);
        return true;
    });
}

#else

void rms_norm_per_block_quant_fp8(torch::Tensor input,
                                  torch::Tensor output_q,
                                  torch::Tensor output_s,
                                  torch::Tensor weight,
                                  int64_t       group_size,
                                  double        rms_eps,
                                  double        quant_eps,
                                  double        min_8bit,
                                  double        max_8bit) {
    (void)input;
    (void)output_q;
    (void)output_s;
    (void)weight;
    (void)group_size;
    (void)rms_eps;
    (void)quant_eps;
    (void)min_8bit;
    (void)max_8bit;
    TORCH_CHECK(false, "rms_norm_per_block_quant_fp8 is only supported on CUDA");
}

#endif

}  // namespace rtp_llm
