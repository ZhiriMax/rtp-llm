// SPDX-License-Identifier: Apache-2.0
// Structure adapted from vLLM's rms_norm_per_block_quant fused op; RTP-LLM
// exposes it as an eager QuantizedActivation producer for SM120 FP8 linear.
#include "rtp_llm/models_py/bindings/cuda/kernels/rms_norm_per_block_quant.h"

#if USING_CUDA
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

__device__ __forceinline__ float blockReduceMax(float val, float* smem) {
    smem[threadIdx.x] = val;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
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
                                              const int   scale_hidden_stride,
                                              const float rms_eps,
                                              const float quant_eps,
                                              const float max_8bit) {
    extern __shared__ float smem[];

    const int token_idx = blockIdx.x;
    const T*  row_input = input + token_idx * hidden_size;

    float local_square_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float val = static_cast<float>(row_input[i]);
        local_square_sum += val * val;
    }
    const float square_sum = blockReduceSum(local_square_sum, smem);
    const float rms        = rsqrtf(square_sum / hidden_size + rms_eps);

    for (int group_idx = 0; group_idx < hidden_groups; ++group_idx) {
        const int group_offset = group_idx * GROUP_SIZE;

        float local_absmax = quant_eps;
        for (int j = threadIdx.x; j < GROUP_SIZE; j += blockDim.x) {
            const int   hidden_idx = group_offset + j;
            const T     val_lowprec = static_cast<T>(static_cast<float>(row_input[hidden_idx]) * rms
                                                 * static_cast<float>(weight[hidden_idx]));
            const float val         = static_cast<float>(val_lowprec);
            local_absmax           = fmaxf(local_absmax, fabsf(val));
        }
        const float absmax  = blockReduceMax(local_absmax, smem);
        const float y_scale = max_8bit / absmax;

        if (threadIdx.x == 0) {
            if constexpr (IS_COLUMN_MAJOR) {
                output_s[group_idx * scale_hidden_stride + token_idx] = absmax / max_8bit;
            } else {
                output_s[token_idx * hidden_groups + group_idx] = absmax / max_8bit;
            }
        }

        for (int j = threadIdx.x; j < GROUP_SIZE; j += blockDim.x) {
            const int   hidden_idx = group_offset + j;
            const T     val_lowprec = static_cast<T>(static_cast<float>(row_input[hidden_idx]) * rms
                                                 * static_cast<float>(weight[hidden_idx]));
            const float val         = static_cast<float>(val_lowprec);
            const float q_val      = fminf(fmaxf(val * y_scale, -max_8bit), max_8bit);
            output_q[token_idx * hidden_size + hidden_idx] = __nv_fp8_e4m3(q_val);
        }
        __syncthreads();
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

    const bool is_column_major     = output_s.stride(0) < output_s.stride(1);
    const int  scale_hidden_stride = static_cast<int>(output_s.stride(1));
    if (is_column_major) {
        TORCH_CHECK(output_s.stride(0) == 1, "column-major output_s must have token stride 1");
        TORCH_CHECK(output_s.stride(1) >= tokens, "column-major output_s hidden stride is too small");
    } else {
        TORCH_CHECK(output_s.stride(1) == 1, "row-major output_s must have hidden stride 1");
        TORCH_CHECK(output_s.stride(0) >= hidden_groups, "row-major output_s token stride is too small");
    }
    const dim3 grid(tokens);
    const dim3 block(256);
    const int  shared_mem_size = block.x * sizeof(float);
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
                scale_hidden_stride,                                                                                   \
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
                scale_hidden_stride,                                                                                   \
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
