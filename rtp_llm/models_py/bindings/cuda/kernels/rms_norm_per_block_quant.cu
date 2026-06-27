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

__device__ __forceinline__ float blockReduceSum(float val, float* smem) {
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

template<typename T, int GROUP_SIZE, bool IS_COLUMN_MAJOR>
__global__ void rmsNormPerBlockQuantFp8Kernel(const T* __restrict__ input,
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
    extern __shared__ float smem[];
    float* reduce_smem  = smem;
    float* group_scales = smem + blockDim.x;

    const int token_idx = blockIdx.x;
    const T*  row_input = input + token_idx * hidden_size;
    constexpr int VEC_SIZE = 4;
    constexpr int GROUP_VEC_SIZE = GROUP_SIZE / VEC_SIZE;
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
    const float square_sum = blockReduceSum(local_square_sum, smem);
    const float rms        = rsqrtf(square_sum / hidden_size + rms_eps);

    int threads_per_group = 1;
    while ((threads_per_group << 1) <= GROUP_VEC_SIZE && (threads_per_group << 1) * hidden_groups <= blockDim.x) {
        threads_per_group <<= 1;
    }
    const int active_thread_size = threads_per_group * hidden_groups;
    if (threadIdx.x < active_thread_size) {
        const int group_idx    = threadIdx.x / threads_per_group;
        const int thread_group = threadIdx.x - group_idx * threads_per_group;
        const int group_offset = group_idx * GROUP_SIZE;

        float local_absmax = quant_eps;
        for (int vec_j = thread_group; vec_j < GROUP_VEC_SIZE; vec_j += threads_per_group) {
            const int hidden_idx = group_offset + vec_j * VEC_SIZE;
            vec_t<T, VEC_SIZE> input_vec;
            vec_t<T, VEC_SIZE> weight_vec;
            input_vec.load(row_input + hidden_idx);
            weight_vec.load(weight + hidden_idx);
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                const float normalized  = static_cast<float>(input_vec[i]) * rms;
                const T     val_lowprec = static_cast<T>(normalized * static_cast<float>(weight_vec[i]));
                const float val         = static_cast<float>(val_lowprec);
                local_absmax            = fmaxf(local_absmax, fabsf(val));
            }
        }
        reduce_smem[threadIdx.x] = local_absmax;
    } else {
        reduce_smem[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = 1; stride < threads_per_group; stride <<= 1) {
        if (threadIdx.x < active_thread_size) {
            const int thread_group = threadIdx.x % threads_per_group;
            if ((thread_group % (stride << 1)) == 0 && thread_group + stride < threads_per_group) {
                reduce_smem[threadIdx.x] = fmaxf(reduce_smem[threadIdx.x], reduce_smem[threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < active_thread_size && threadIdx.x % threads_per_group == 0) {
        const int   group_idx = threadIdx.x / threads_per_group;
        const float absmax    = reduce_smem[threadIdx.x];
        group_scales[group_idx] = max_8bit / absmax;
        if constexpr (IS_COLUMN_MAJOR) {
            output_s[group_idx * scale_hidden_stride + token_idx] = absmax / max_8bit;
        } else {
            output_s[token_idx * scale_token_stride + group_idx] = absmax / max_8bit;
        }
    }
    __syncthreads();

    auto output_pair = reinterpret_cast<__nv_fp8x2_storage_t*>(output_q + token_idx * hidden_size);
    for (int vec_idx = threadIdx.x; vec_idx < vec_hidden_size; vec_idx += blockDim.x) {
        const int hidden_idx = vec_idx * VEC_SIZE;
        vec_t<T, VEC_SIZE> input_vec;
        vec_t<T, VEC_SIZE> weight_vec;
        input_vec.load(row_input + hidden_idx);
        weight_vec.load(weight + hidden_idx);
        const float scale = group_scales[hidden_idx / GROUP_SIZE];

        float output_values[VEC_SIZE];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            const float normalized  = static_cast<float>(input_vec[i]) * rms;
            const T     val_lowprec = static_cast<T>(normalized * static_cast<float>(weight_vec[i]));
            output_values[i]        = fminf(fmaxf(static_cast<float>(val_lowprec) * scale, -max_8bit), max_8bit);
        }

        const float2 outputx2_0      = {output_values[0], output_values[1]};
        const float2 outputx2_1      = {output_values[2], output_values[3]};
        output_pair[vec_idx * 2]     = __nv_cvt_float2_to_fp8x2(outputx2_0, __NV_SATFINITE, __NV_E4M3);
        output_pair[vec_idx * 2 + 1] = __nv_cvt_float2_to_fp8x2(outputx2_1, __NV_SATFINITE, __NV_E4M3);
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
    TORCH_CHECK(quant_eps > 0.0, "quant_eps must be positive");
    TORCH_CHECK(input.size(1) % 4 == 0, "hidden size must be divisible by 4");
    TORCH_CHECK(group_size % 4 == 0, "group_size must be divisible by 4");
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 || group_size == 128,
                "Unsupported group_size");

    const int tokens        = static_cast<int>(input.size(0));
    const int hidden_size   = static_cast<int>(input.size(1));
    const int hidden_groups = static_cast<int>(hidden_size / group_size);
    TORCH_CHECK(tokens > 0, "input must have at least one token");
    TORCH_CHECK(hidden_size > 0 && hidden_size % group_size == 0, "hidden size must be divisible by group_size");
    TORCH_CHECK(output_s.size(0) == tokens && output_s.size(1) == hidden_groups,
                "output_s shape must be (tokens, hidden_size / group_size)");
    TORCH_CHECK(output_s.numel() >= tokens * hidden_groups, "output_s buffer is too small");

    const bool is_column_major    = output_s.stride(0) < output_s.stride(1);
    const int  scale_token_stride = static_cast<int>(output_s.stride(0));
    const int  scale_group_stride = static_cast<int>(output_s.stride(1));
    if (is_column_major) {
        TORCH_CHECK(output_s.stride(0) == 1, "column-major output_s must have token stride 1");
        TORCH_CHECK(output_s.stride(1) >= tokens, "column-major output_s hidden stride is too small");
    } else {
        TORCH_CHECK(output_s.stride(1) == 1, "row-major output_s must have hidden stride 1");
        TORCH_CHECK(output_s.stride(0) >= hidden_groups, "row-major output_s token stride is too small");
    }
    const int  block_size = hidden_groups > 512 ? 1024 : (hidden_groups > 256 ? 512 : 256);
    TORCH_CHECK(hidden_groups <= block_size, "hidden group count is too large for fused RMSNorm quant");
    const dim3 grid(tokens);
    const dim3 block(block_size);
    const int  shared_mem_size = (block.x + hidden_groups) * sizeof(float);
    cudaStream_t stream        = at::cuda::getCurrentCUDAStream();

#define LAUNCH_RMS_NORM_PER_BLOCK_QUANT(GROUP_SIZE, T)                                                                \
    do {                                                                                                               \
        if (is_column_major) {                                                                                         \
            rmsNormPerBlockQuantFp8Kernel<T, GROUP_SIZE, true><<<grid, block, shared_mem_size, stream>>>(              \
                static_cast<T*>(input.data_ptr()),                                                                     \
                reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),                                                 \
                static_cast<float*>(output_s.data_ptr()),                                                              \
                static_cast<T*>(weight.data_ptr()),                                                                    \
                hidden_size,                                                                                           \
                hidden_groups,                                                                                         \
                scale_token_stride,                                                                                    \
                scale_group_stride,                                                                                    \
                static_cast<float>(rms_eps),                                                                           \
                static_cast<float>(quant_eps),                                                                         \
                static_cast<float>(max_8bit));                                                                         \
        } else {                                                                                                       \
            rmsNormPerBlockQuantFp8Kernel<T, GROUP_SIZE, false><<<grid, block, shared_mem_size, stream>>>(             \
                static_cast<T*>(input.data_ptr()),                                                                     \
                reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),                                                 \
                static_cast<float*>(output_s.data_ptr()),                                                              \
                static_cast<T*>(weight.data_ptr()),                                                                    \
                hidden_size,                                                                                           \
                hidden_groups,                                                                                         \
                scale_token_stride,                                                                                    \
                scale_group_stride,                                                                                    \
                static_cast<float>(rms_eps),                                                                           \
                static_cast<float>(quant_eps),                                                                         \
                static_cast<float>(max_8bit));                                                                         \
        }                                                                                                              \
    } while (0)

#define LAUNCH_RMS_NORM_PER_BLOCK_QUANT_OUTER(T)                                                                       \
    do {                                                                                                               \
        switch (group_size) {                                                                                          \
            case 16:                                                                                                   \
                LAUNCH_RMS_NORM_PER_BLOCK_QUANT(16, T);                                                                \
                break;                                                                                                 \
            case 32:                                                                                                   \
                LAUNCH_RMS_NORM_PER_BLOCK_QUANT(32, T);                                                                \
                break;                                                                                                 \
            case 64:                                                                                                   \
                LAUNCH_RMS_NORM_PER_BLOCK_QUANT(64, T);                                                                \
                break;                                                                                                 \
            case 128:                                                                                                  \
                LAUNCH_RMS_NORM_PER_BLOCK_QUANT(128, T);                                                               \
                break;                                                                                                 \
            default:                                                                                                   \
                TORCH_CHECK(false, "Unsupported group_size");                                                          \
        }                                                                                                              \
    } while (0)

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
        LAUNCH_RMS_NORM_PER_BLOCK_QUANT_OUTER(scalar_t);
        return true;
    });

#undef LAUNCH_RMS_NORM_PER_BLOCK_QUANT_OUTER
#undef LAUNCH_RMS_NORM_PER_BLOCK_QUANT
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
