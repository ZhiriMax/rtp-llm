// SPDX-License-Identifier: Apache-2.0
//
// Experimental gated MLP producer boundaries for SM120 FP8.
//
// This file contains three milestones for the article-style gated MLP kernel:
// 1. staged producer: input quantization, ordinary SM120 gate/up GEMM, then
//    existing fused silu/mul FP8 quantization. It proves the ABI but still
//    materializes BF16 [M, 2I].
// 2. semi-tail emulation: ordinary gate/up GEMM, then BF16 activated [M, I]
//    plus existing non-fused quantization. It measures the planned tail.
// 3. reference producer: a producer-specific tiled CUDA kernel directly writes
//    FP8 activated [M, I] plus dynamic scales. It proves the final output
//    contract without BF16 [M, 2I], but is not the final SM120 tensor-MMA performance
//    implementation.

#include "cutlass_gated_mlp_producer_sm120_fp8.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdlib>
#include <cstring>

#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_8bit.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_8bit_v2.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "cutlass_scaled_mm_blockwise_sm120_fp8.h"

namespace {

constexpr int     kGroupSize = 128;
constexpr double  kFp8Min    = -448.0;
constexpr double  kFp8Max    = 448.0;
constexpr double  kInputEps  = 1e-4;
constexpr bool    kScaleUe8m0 = false;

struct GatedMlpProblem {
    int64_t m;
    int64_t h;
    int64_t n2;
    int64_t i;
};

struct ArticleMmaProducerTarget {
    // This is the intended article-style performance shape, not the CUDA
    // reference CTA shape below. Two math warpgroups each own 64 output
    // columns (up + gate interleaved in one 128-column MMA tile), while one
    // producer warpgroup feeds the shared A tile plus the two paired B tiles.
    static constexpr int k_m                  = 64;
    static constexpr int k_n                  = kGroupSize;
    static constexpr int k_pair_n             = 64;
    static constexpr int k_k                  = kGroupSize;
    static constexpr int k_warpgroup_threads = 128;
    static constexpr int k_math_warpgroups    = 2;
    static constexpr int k_tma_warpgroups     = 1;
    static constexpr int k_math_threads       = k_math_warpgroups * k_warpgroup_threads;
    static constexpr int k_cta_threads        = (k_math_warpgroups + k_tma_warpgroups) * k_warpgroup_threads;
};

struct ArticleMmaProducerTile {
    static constexpr int k_m             = ArticleMmaProducerTarget::k_m;
    static constexpr int k_n             = ArticleMmaProducerTarget::k_n;
    static constexpr int k_pair_n        = ArticleMmaProducerTarget::k_pair_n;
    static constexpr int k_pairs         = k_n / k_pair_n;
    static constexpr int k_k             = ArticleMmaProducerTarget::k_k;
    static constexpr int k_mma_k         = 32;
    static constexpr int k_mma_steps     = k_k / k_mma_k;
    static constexpr int k_math_threads  = ArticleMmaProducerTarget::k_math_threads;
    static constexpr int k_tma_threads   = ArticleMmaProducerTarget::k_warpgroup_threads;
    static constexpr int k_cta_threads   = ArticleMmaProducerTarget::k_cta_threads;
    static constexpr int k_acc_registers = k_n / 2;
};

struct CudaReferenceProducerTile {
    // Validation-only scalar CUDA tile. It mirrors the article's M/N/K output
    // grouping and paired B layout, but maps rows/columns to ordinary CUDA
    // threads instead of using the 384-thread warp-specialized MMA schedule.
    static constexpr int k_m             = ArticleMmaProducerTarget::k_m;
    static constexpr int k_n             = ArticleMmaProducerTarget::k_n;
    static constexpr int k_pair_n        = ArticleMmaProducerTarget::k_pair_n;
    static constexpr int k_pairs         = k_n / k_pair_n;
    static constexpr int k_k             = ArticleMmaProducerTarget::k_k;
    static constexpr int k_threads_x     = 16;
    static constexpr int k_cols_per_lane = k_n / k_threads_x;
    static constexpr int k_threads       = k_m * k_threads_x;
    static constexpr int k_smem_a_bytes  = k_m * k_k;
    static constexpr int k_smem_b_bytes  = k_pairs * k_n * k_k;
    static constexpr int k_smem_abs_bytes = k_m * k_threads_x * static_cast<int>(sizeof(float));
    static constexpr int k_smem_bytes     = k_smem_a_bytes + k_smem_b_bytes + k_smem_abs_bytes;
};

static_assert(ArticleMmaProducerTarget::k_n == kGroupSize);
static_assert(ArticleMmaProducerTarget::k_n == 2 * ArticleMmaProducerTarget::k_pair_n);
static_assert(ArticleMmaProducerTarget::k_cta_threads == 384);
static_assert(ArticleMmaProducerTarget::k_math_threads == 256);
static_assert(ArticleMmaProducerTile::k_mma_steps == 4);
static_assert(ArticleMmaProducerTile::k_acc_registers == 64);
static_assert(ArticleMmaProducerTile::k_cta_threads == 384);
static_assert(CudaReferenceProducerTile::k_m == ArticleMmaProducerTarget::k_m);
static_assert(CudaReferenceProducerTile::k_n == ArticleMmaProducerTarget::k_n);
static_assert(CudaReferenceProducerTile::k_threads <= 1024);
static_assert(CudaReferenceProducerTile::k_smem_bytes == 45056);

template<typename Tile>
struct PairedAccumulator {
    float paired[Tile::k_cols_per_lane][2];

    __device__ __forceinline__ void clear() {
        for (int c = 0; c < Tile::k_cols_per_lane; ++c) {
            paired[c][0] = 0.0f;
            paired[c][1] = 0.0f;
        }
    }

    __device__ __forceinline__ float& up(int c) {
        return paired[c][0];
    }

    __device__ __forceinline__ float& gate(int c) {
        return paired[c][1];
    }
};

struct PairedBodyContext {
    float const* weight_scale;
    float        a_scale;
    int          lane_m;
    int          lane_n;
    int64_t      row;
    int64_t      m;
    int64_t      output_group;
    int64_t      k_group;
    int64_t      scale_k;
    int64_t      intermediate_size;
};

template<typename Tile>
__device__ __forceinline__ int pair_index(int local_col) {
    return local_col / Tile::k_pair_n;
}

template<typename Tile>
__device__ __forceinline__ int pair_col_index(int local_col) {
    return local_col - pair_index<Tile>(local_col) * Tile::k_pair_n;
}

__device__ __forceinline__ int64_t gate_scale_group(int64_t output_group) {
    return output_group;
}

__device__ __forceinline__ int64_t up_scale_group(int64_t output_group, int64_t intermediate_size) {
    return output_group + intermediate_size / kGroupSize;
}

struct PairedScaleCoeff {
    float gate;
    float up;
};

__device__ __forceinline__ PairedScaleCoeff load_paired_scale_coeff(PairedBodyContext const& ctx) {
    float gate_scale = ctx.weight_scale[gate_scale_group(ctx.output_group) * ctx.scale_k + ctx.k_group];
    float up_scale   = ctx.weight_scale[up_scale_group(ctx.output_group, ctx.intermediate_size) * ctx.scale_k
                                      + ctx.k_group];
    return PairedScaleCoeff{ctx.a_scale * gate_scale, ctx.a_scale * up_scale};
}

void check_cuda_same_device(torch::Tensor const& tensor, char const* name, c10::Device const& device) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device,
                name,
                " must be on the same device as input, got ",
                tensor.device(),
                " vs ",
                device);
}

void check_contiguous(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

torch::Tensor column_major_scale(torch::Tensor const& like, int64_t groups, int64_t tokens) {
    auto storage = torch::empty({groups, tokens}, like.options().dtype(torch::kFloat32));
    return storage.transpose(0, 1);
}

bool use_v2_input_quant() {
    char const* value = std::getenv("FP8_GROUP_QUANT_KERNEL");
    return value != nullptr && std::strcmp(value, "v2") == 0;
}

__device__ __forceinline__ float silu(float val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    float half = 0.5f * val;
    return half * (1.0f + __tanhf(half));
#else
    return val / (1.0f + __expf(-val));
#endif
}

__device__ __forceinline__ __nv_bfloat16 staged_equivalent_activation(float gate, float up) {
    __nv_bfloat16 gate_bf16 = __float2bfloat16(gate);
    __nv_bfloat16 up_bf16   = __float2bfloat16(up);
    __nv_bfloat16 silu_bf16 = __float2bfloat16(silu(__bfloat162float(gate_bf16)));
    return __float2bfloat16(__bfloat162float(silu_bf16) * __bfloat162float(up_bf16));
}

template<typename Tile>
__device__ __forceinline__ void load_a_tile(__nv_fp8_e4m3 const* __restrict__ input,
                                            __nv_fp8_e4m3 (&a_tile)[Tile::k_m][Tile::k_k],
                                            int                               linear,
                                            int64_t                           row_base,
                                            int64_t                           k_base,
                                            int64_t                           m,
                                            int64_t                           h) {
    for (int load_idx = linear; load_idx < Tile::k_m * Tile::k_k; load_idx += Tile::k_threads) {
        int     local_m = load_idx / Tile::k_k;
        int     local_k = load_idx - local_m * Tile::k_k;
        int64_t src_row = row_base + local_m;
        int64_t src_k   = k_base + local_k;
        auto    value   = __nv_fp8_e4m3(0.0f);
        if (src_row < m) {
            value = input[src_row * h + src_k];
        }
        a_tile[local_m][local_k] = value;
    }
}

template<typename Tile>
__device__ __forceinline__ void load_paired_b_tile(
    __nv_fp8_e4m3 const* __restrict__ weight,
    __nv_fp8_e4m3 (&paired_b_tile)[Tile::k_pairs][Tile::k_n][Tile::k_k],
    int     linear,
    int64_t output_group,
    int64_t k_base,
    int64_t intermediate_size,
    int64_t h) {
    for (int load_idx = linear; load_idx < Tile::k_n * Tile::k_k; load_idx += Tile::k_threads) {
        int     local_col = load_idx / Tile::k_k;
        int     local_k   = load_idx - local_col * Tile::k_k;
        int64_t col       = output_group * Tile::k_n + local_col;
        int64_t k         = k_base + local_k;
        int     pair      = pair_index<Tile>(local_col);
        int     pair_col  = pair_col_index<Tile>(local_col);
        paired_b_tile[pair][pair_col][local_k]                  = weight[(col + intermediate_size) * h + k];
        paired_b_tile[pair][Tile::k_pair_n + pair_col][local_k] = weight[col * h + k];
    }
}

template<typename Tile>
struct CudaReferencePairedBody {
    using TileShape = Tile;
    static constexpr bool k_uses_tensor_mma = false;

    // Reference body for the future tensor-MMA cut: accumulate raw FP8 A/B products
    // for one K tile, then apply the input and weight scales before adding to
    // the final paired gate/up accumulator.
    __device__ __forceinline__ static void run(
        __nv_fp8_e4m3 const (&a_tile)[Tile::k_m][Tile::k_k],
        __nv_fp8_e4m3 const (&paired_b_tile)[Tile::k_pairs][Tile::k_n][Tile::k_k],
        PairedAccumulator<Tile>& acc,
        PairedBodyContext const& ctx) {
        if (ctx.row >= ctx.m) {
            return;
        }

        PairedScaleCoeff coeff = load_paired_scale_coeff(ctx);
        for (int local_k = 0; local_k < Tile::k_k; ++local_k) {
            float a = rtp_llm::cuda_cast<float>(a_tile[ctx.lane_m][local_k]);
            for (int c = 0; c < Tile::k_cols_per_lane; ++c) {
                int local_col = ctx.lane_n * Tile::k_cols_per_lane + c;
                int pair      = pair_index<Tile>(local_col);
                int pair_col  = pair_col_index<Tile>(local_col);
                acc.up(c) += a * rtp_llm::cuda_cast<float>(paired_b_tile[pair][pair_col][local_k]) * coeff.up;
                acc.gate(c) += a
                               * rtp_llm::cuda_cast<float>(
                                   paired_b_tile[pair][Tile::k_pair_n + pair_col][local_k])
                               * coeff.gate;
            }
        }
    }
};

struct ArticleMmaPairedBody {
    using TileShape = ArticleMmaProducerTile;
    static constexpr bool k_uses_tensor_mma = true;
    static constexpr int  k_acc_registers   = TileShape::k_acc_registers;
};

static_assert(ArticleMmaPairedBody::TileShape::k_acc_registers == ArticleMmaPairedBody::k_acc_registers);

template<typename Tile>
__device__ __forceinline__ void store_quantized_activation(__nv_bfloat16 const* __restrict__ bias,
                                                           __nv_fp8_e4m3* __restrict__       data,
                                                           float* __restrict__               scale,
                                                           float (&row_abs)[Tile::k_m][Tile::k_threads_x],
                                                           PairedAccumulator<Tile>& acc,
                                                           int     lane_m,
                                                           int     lane_n,
                                                           int64_t row,
                                                           int64_t col_base,
                                                           int64_t output_group,
                                                           int64_t m,
                                                           int64_t intermediate_size) {
    float activated[Tile::k_cols_per_lane] = {};
    float local_abs                        = static_cast<float>(kInputEps);
    if (row < m) {
        for (int c = 0; c < Tile::k_cols_per_lane; ++c) {
            int64_t col = col_base + c;
            if (bias != nullptr) {
                acc.gate(c) += __bfloat162float(bias[col]);
                acc.up(c) += __bfloat162float(bias[col + intermediate_size]);
            }
            activated[c] = __bfloat162float(staged_equivalent_activation(acc.gate(c), acc.up(c)));
            local_abs    = fmaxf(local_abs, fabsf(activated[c]));
        }
    }
    row_abs[lane_m][lane_n] = local_abs;
    __syncthreads();

    for (int offset = Tile::k_threads_x / 2; offset > 0; offset >>= 1) {
        if (lane_n < offset) {
            row_abs[lane_m][lane_n] = fmaxf(row_abs[lane_m][lane_n], row_abs[lane_m][lane_n + offset]);
        }
        __syncthreads();
    }

    float y_scale_inv = row_abs[lane_m][0] / static_cast<float>(kFp8Max);
    float y_scale     = static_cast<float>(kFp8Max) / row_abs[lane_m][0];
    if (row < m && lane_n == 0) {
        scale[output_group * m + row] = y_scale_inv;
    }
    if (row < m) {
        for (int c = 0; c < Tile::k_cols_per_lane; ++c) {
            int64_t col   = col_base + c;
            float   q_val = activated[c] * y_scale;
            q_val         = fminf(fmaxf(q_val, static_cast<float>(kFp8Min)), static_cast<float>(kFp8Max));
            data[row * intermediate_size + col] = __nv_fp8_e4m3(q_val);
        }
    }
}

__global__ void activate_gate_up_bf16_kernel(__nv_bfloat16 const* __restrict__ gate_up,
                                             __nv_bfloat16* __restrict__       activated,
                                             int64_t                           elements,
                                             int64_t                           i) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int64_t row  = idx / i;
    int64_t col  = idx - row * i;
    int64_t base = row * i * 2 + col;
    float   gate = __bfloat162float(gate_up[base]);
    float   up   = __bfloat162float(gate_up[base + i]);
    activated[idx] = staged_equivalent_activation(gate, up);
}

template<typename Body>
__global__ void gate_up_full_output_producer_kernel(__nv_fp8_e4m3 const* __restrict__ input,
                                                    float const* __restrict__         input_scale,
                                                    __nv_fp8_e4m3 const* __restrict__ weight,
                                                    float const* __restrict__         weight_scale,
                                                    __nv_bfloat16 const* __restrict__ bias,
                                                    __nv_fp8_e4m3* __restrict__       data,
                                                    float* __restrict__               scale,
                                                    int64_t                           m,
                                                    int64_t                           h,
                                                    int64_t                           i,
                                                    int64_t                           scale_k) {
    using Tile = typename Body::TileShape;

    int64_t row_base = static_cast<int64_t>(blockIdx.x) * Tile::k_m;
    int64_t group    = blockIdx.y;
    int     lane_m   = threadIdx.y;
    int     lane_n   = threadIdx.x;
    int     linear   = lane_m * Tile::k_threads_x + lane_n;
    int64_t row      = row_base + lane_m;
    int64_t col_base = group * Tile::k_n + lane_n * Tile::k_cols_per_lane;

    __shared__ __nv_fp8_e4m3 a_tile[Tile::k_m][Tile::k_k];
    __shared__ __nv_fp8_e4m3 paired_b_tile[Tile::k_pairs][Tile::k_n][Tile::k_k];
    __shared__ float         row_abs[Tile::k_m][Tile::k_threads_x];

    PairedAccumulator<Tile> acc;
    acc.clear();

    for (int64_t k_base = 0; k_base < h; k_base += Tile::k_k) {
        int64_t k_group = k_base / Tile::k_k;
        float   a_scale = row < m ? input_scale[k_group * m + row] : 0.0f;
        load_a_tile<Tile>(input, a_tile, linear, row_base, k_base, m, h);
        load_paired_b_tile<Tile>(weight, paired_b_tile, linear, group, k_base, i, h);
        __syncthreads();

        PairedBodyContext ctx{weight_scale, a_scale, lane_m, lane_n, row, m, group, k_group, scale_k, i};
        Body::run(a_tile, paired_b_tile, acc, ctx);
        __syncthreads();
    }

    store_quantized_activation<Tile>(bias, data, scale, row_abs, acc, lane_m, lane_n, row, col_base, group, m, i);
}

GatedMlpProblem check_problem(torch::Tensor const& input,
                              torch::Tensor const& gate_up_weight,
                              torch::Tensor const& gate_up_scale) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    auto device = input.device();
    check_cuda_same_device(gate_up_weight, "gate_up_weight", device);
    check_cuda_same_device(gate_up_scale, "gate_up_scale", device);

    check_contiguous(input, "input");
    check_contiguous(gate_up_weight, "gate_up_weight");

    TORCH_CHECK(input.dtype() == at::ScalarType::BFloat16, "input must be bfloat16, got ", input.dtype());
    TORCH_CHECK(gate_up_weight.dtype() == torch::kFloat8_e4m3fn,
                "gate_up_weight must be float8_e4m3fn, got ",
                gate_up_weight.dtype());
    TORCH_CHECK(gate_up_scale.dtype() == torch::kFloat32,
                "gate_up_scale must be float32, got ",
                gate_up_scale.dtype());
    TORCH_CHECK(input.dim() == 2, "input must be 2D, got ", input.dim(), "D");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be 2D, got ", gate_up_weight.dim(), "D");
    TORCH_CHECK(gate_up_scale.dim() == 2, "gate_up_scale must be 2D, got ", gate_up_scale.dim(), "D");

    GatedMlpProblem p{input.size(0), input.size(1), gate_up_weight.size(0), gate_up_weight.size(0) / 2};
    TORCH_CHECK(p.h % kGroupSize == 0, "input hidden size must be divisible by 128, got ", p.h);
    TORCH_CHECK(p.n2 % 2 == 0, "gate_up_weight output dimension must be even, got ", p.n2);
    TORCH_CHECK(gate_up_weight.size(1) == p.h,
                "gate_up_weight shape must be (2I, H), got (",
                gate_up_weight.size(0),
                ", ",
                gate_up_weight.size(1),
                ") for H=",
                p.h);
    TORCH_CHECK(p.i % ArticleMmaProducerTarget::k_n == 0,
                "intermediate size must be divisible by producer output group ",
                ArticleMmaProducerTarget::k_n,
                ", got ",
                p.i);
    TORCH_CHECK(gate_up_scale.size(0) == ceil_div(p.n2, kGroupSize)
                    && gate_up_scale.size(1) == ceil_div(p.h, kGroupSize),
                "gate_up_scale shape must be (ceil_div(2I, 128), ceil_div(H, 128)), got (",
                gate_up_scale.size(0),
                ", ",
                gate_up_scale.size(1),
                ")");
    TORCH_CHECK(gate_up_scale.stride(0) == ceil_div(p.h, kGroupSize) && gate_up_scale.stride(1) == 1,
                "gate_up_scale must use K-major contiguous scale layout with stride "
                "(ceil_div(H, 128), 1), got (",
                gate_up_scale.stride(0),
                ", ",
                gate_up_scale.stride(1),
                ")");
    return p;
}

std::optional<torch::Tensor> normalize_bias(std::optional<torch::Tensor> const& bias,
                                            c10::Device const&                  device,
                                            at::ScalarType                      dtype,
                                            int64_t                             n2) {
    if (!bias.has_value()) {
        return std::nullopt;
    }
    check_cuda_same_device(*bias, "gate_up_bias", device);
    TORCH_CHECK(bias->dtype() == dtype, "gate_up_bias dtype must match input dtype");
    TORCH_CHECK(bias->dim() == 1 || bias->dim() == 2,
                "gate_up_bias must be 1D [2I] or 2D [1, 2I], got ",
                bias->dim(),
                "D");
    TORCH_CHECK(bias->dim() == 1 || bias->size(0) == 1,
                "2D gate_up_bias first dimension must be 1, got ",
                bias->size(0));
    TORCH_CHECK(bias->numel() == n2, "gate_up_bias must have 2I (", n2, ") elements, got ", bias->numel());
    return bias->contiguous();
}

std::tuple<torch::Tensor, torch::Tensor> quantize_input(torch::Tensor const& input, GatedMlpProblem const& p) {
    auto input_fp8    = torch::empty({p.m, p.h}, input.options().dtype(torch::kFloat8_e4m3fn));
    auto input_scales = column_major_scale(input, p.h / kGroupSize, p.m);
    if (use_v2_input_quant()) {
        rtp_llm::sgl_per_token_group_quant_8bit_v2(input,
                                                   input_fp8,
                                                   input_scales,
                                                   kGroupSize,
                                                   kInputEps,
                                                   kFp8Min,
                                                   kFp8Max,
                                                   kScaleUe8m0,
                                                   false,
                                                   std::nullopt);
    } else {
        rtp_llm::per_token_group_quant_8bit(
            input, input_fp8, input_scales, kGroupSize, kInputEps, kFp8Min, kFp8Max, kScaleUe8m0);
    }
    return std::make_tuple(input_fp8, input_scales);
}

std::tuple<torch::Tensor, torch::Tensor> empty_quantized_activation(torch::Tensor const&   like,
                                                                    GatedMlpProblem const& p) {
    auto data  = torch::empty({p.m, p.i}, like.options().dtype(torch::kFloat8_e4m3fn));
    auto scale = column_major_scale(like, p.i / kGroupSize, p.m);
    return std::make_tuple(data, scale);
}

torch::Tensor compute_staged_gate_up(torch::Tensor const&                input_fp8,
                                     torch::Tensor const&                input_scales,
                                     torch::Tensor const&                gate_up_weight,
                                     torch::Tensor const&                gate_up_scale,
                                     std::optional<torch::Tensor> const& bias,
                                     torch::Tensor const&                like,
                                     GatedMlpProblem const&              p) {
    auto gate_up = torch::empty({p.m, p.n2}, like.options());
    cutlass_scaled_mm_blockwise_sm120_fp8(gate_up, input_fp8, gate_up_weight, input_scales, gate_up_scale, bias);
    return gate_up;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_staged_gate_up(torch::Tensor const& gate_up,
                                                                 torch::Tensor const& like,
                                                                 GatedMlpProblem const& p) {
    auto data  = torch::empty({p.m, p.i}, like.options().dtype(torch::kFloat8_e4m3fn));
    auto scale = column_major_scale(like, p.i / kGroupSize, p.m);
    rtp_llm::sgl_per_token_group_quant_8bit_v2(gate_up,
                                               data,
                                               scale,
                                               kGroupSize,
                                               kInputEps,
                                               kFp8Min,
                                               kFp8Max,
                                               kScaleUe8m0,
                                               true,
                                               std::nullopt);
    return std::make_tuple(data, scale);
}

torch::Tensor activate_staged_gate_up(torch::Tensor const& gate_up,
                                      torch::Tensor const& like,
                                      GatedMlpProblem const& p) {
    auto activated = torch::empty({p.m, p.i}, like.options());
    if (p.m == 0) {
        return activated;
    }

    constexpr int threads  = 256;
    int64_t       elements = p.m * p.i;
    auto          blocks   = static_cast<unsigned int>((elements + threads - 1) / threads);
    dim3          grid(blocks);
    activate_gate_up_bf16_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<__nv_bfloat16 const*>(gate_up.const_data_ptr()),
        static_cast<__nv_bfloat16*>(activated.data_ptr()),
        elements,
        p.i);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return activated;
}

torch::Tensor compute_semi_tail_emulation(torch::Tensor const&                input_fp8,
                                          torch::Tensor const&                input_scales,
                                          torch::Tensor const&                gate_up_weight,
                                          torch::Tensor const&                gate_up_scale,
                                          std::optional<torch::Tensor> const& bias,
                                          torch::Tensor const&                like,
                                          GatedMlpProblem const&              p) {
    auto gate_up = compute_staged_gate_up(input_fp8, input_scales, gate_up_weight, gate_up_scale, bias, like, p);
    return activate_staged_gate_up(gate_up, like, p);
}

struct FullOutputProducerInputs {
    torch::Tensor const&                input_fp8;
    torch::Tensor const&                input_scales;
    torch::Tensor const&                gate_up_weight;
    torch::Tensor const&                gate_up_scale;
    std::optional<torch::Tensor> const& bias;
    torch::Tensor const&                like;
    GatedMlpProblem const&              problem;
};

template<typename Body>
std::tuple<torch::Tensor, torch::Tensor>
compute_cuda_reference_full_output_producer(FullOutputProducerInputs const& inputs) {
    auto const& p             = inputs.problem;
    auto [data, scale]        = empty_quantized_activation(inputs.like, p);
    if (p.m == 0) {
        return std::make_tuple(data, scale);
    }

    using Tile = typename Body::TileShape;
    dim3 grid(static_cast<unsigned int>(ceil_div(p.m, Tile::k_m)), static_cast<unsigned int>(p.i / Tile::k_n));
    dim3 block(Tile::k_threads_x, Tile::k_m);
    auto bias_ptr =
        inputs.bias.has_value() ? static_cast<__nv_bfloat16 const*>(inputs.bias->const_data_ptr()) : nullptr;
    static_assert(!Body::k_uses_tensor_mma, "Tensor-MMA producer bodies need a SM120-specific launcher.");
    gate_up_full_output_producer_kernel<Body><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<__nv_fp8_e4m3 const*>(inputs.input_fp8.const_data_ptr()),
        static_cast<float const*>(inputs.input_scales.const_data_ptr()),
        static_cast<__nv_fp8_e4m3 const*>(inputs.gate_up_weight.const_data_ptr()),
        static_cast<float const*>(inputs.gate_up_scale.const_data_ptr()),
        bias_ptr,
        static_cast<__nv_fp8_e4m3*>(data.data_ptr()),
        static_cast<float*>(scale.data_ptr()),
        p.m,
        p.h,
        p.i,
        p.h / kGroupSize);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(data, scale);
}

std::tuple<torch::Tensor, torch::Tensor>
compute_tiled_reference_producer(FullOutputProducerInputs const& inputs) {
    using Body = CudaReferencePairedBody<CudaReferenceProducerTile>;
    return compute_cuda_reference_full_output_producer<Body>(inputs);
}

std::tuple<torch::Tensor, torch::Tensor>
run_full_output_producer(FullOutputProducerInputs const& inputs) {
    // Keep the external op contract stable while the producer body evolves.
    // The current body is a tiled CUDA reference. The performance body
    // should replace only this call with article-style SM120 tensor-MMA producer/mainloop:
    // paired gate/up B tiles, shared A tile, in-register activation, online
    // 128-column amax, and direct FP8 data+scale stores.
    return compute_tiled_reference_producer(inputs);
}

std::tuple<torch::Tensor, torch::Tensor>
quantize_activated_tail(torch::Tensor const& activated, torch::Tensor const& like, GatedMlpProblem const& p) {
    auto data  = torch::empty({p.m, p.i}, like.options().dtype(torch::kFloat8_e4m3fn));
    auto scale = column_major_scale(like, p.i / kGroupSize, p.m);
    if (use_v2_input_quant()) {
        rtp_llm::sgl_per_token_group_quant_8bit_v2(activated,
                                                   data,
                                                   scale,
                                                   kGroupSize,
                                                   kInputEps,
                                                   kFp8Min,
                                                   kFp8Max,
                                                   kScaleUe8m0,
                                                   false,
                                                   std::nullopt);
    } else {
        rtp_llm::per_token_group_quant_8bit(
            activated, data, scale, kGroupSize, kInputEps, kFp8Min, kFp8Max, kScaleUe8m0);
    }
    return std::make_tuple(data, scale);
}

using ActivatedProducer = torch::Tensor (*)(torch::Tensor const&,
                                            torch::Tensor const&,
                                            torch::Tensor const&,
                                            torch::Tensor const&,
                                            std::optional<torch::Tensor> const&,
                                            torch::Tensor const&,
                                            GatedMlpProblem const&);

std::tuple<torch::Tensor, torch::Tensor>
run_activated_producer(torch::Tensor const&                input,
                       torch::Tensor const&                gate_up_weight,
                       torch::Tensor const&                gate_up_scale,
                       std::optional<torch::Tensor> const& bias,
                       GatedMlpProblem const&              p,
                       ActivatedProducer                   producer) {
    if (p.m == 0) {
        return empty_quantized_activation(input, p);
    }

    auto [input_fp8, input_scales] = quantize_input(input, p);
    auto activated = producer(input_fp8, input_scales, gate_up_weight, gate_up_scale, bias, input, p);
    return quantize_activated_tail(activated, input, p);
}

std::tuple<torch::Tensor, torch::Tensor>
run_reference_producer(torch::Tensor const&                input,
                       torch::Tensor const&                gate_up_weight,
                       torch::Tensor const&                gate_up_scale,
                       std::optional<torch::Tensor> const& bias,
                       GatedMlpProblem const&              p) {
    if (p.m == 0) {
        return empty_quantized_activation(input, p);
    }

    auto [input_fp8, input_scales] = quantize_input(input, p);
    FullOutputProducerInputs producer_inputs{
        input_fp8, input_scales, gate_up_weight, gate_up_scale, bias, input, p};
    return run_full_output_producer(producer_inputs);
}

std::tuple<torch::Tensor, torch::Tensor>
run_staged_producer(torch::Tensor const&                input,
                    torch::Tensor const&                gate_up_weight,
                    torch::Tensor const&                gate_up_scale,
                    std::optional<torch::Tensor> const& bias,
                    GatedMlpProblem const&              p) {
    if (p.m == 0) {
        return empty_quantized_activation(input, p);
    }

    auto [input_fp8, input_scales] = quantize_input(input, p);
    auto gate_up = compute_staged_gate_up(input_fp8, input_scales, gate_up_weight, gate_up_scale, bias, input, p);
    return quantize_staged_gate_up(gate_up, input, p);
}

std::tuple<torch::Tensor, torch::Tensor>
run_semi_tail_emulation(torch::Tensor const&                input,
                        torch::Tensor const&                gate_up_weight,
                        torch::Tensor const&                gate_up_scale,
                        std::optional<torch::Tensor> const& bias,
                        GatedMlpProblem const&              p) {
    return run_activated_producer(input, gate_up_weight, gate_up_scale, bias, p, compute_semi_tail_emulation);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
cutlass_gated_mlp_staged_producer_sm120_fp8(torch::Tensor const&                input,
                                            torch::Tensor const&                gate_up_weight,
                                            torch::Tensor const&                gate_up_scale,
                                            std::optional<torch::Tensor> const& gate_up_bias) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto problem  = check_problem(input, gate_up_weight, gate_up_scale);
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    auto bias_arg = normalize_bias(gate_up_bias, input.device(), input.dtype(), problem.n2);
    return run_staged_producer(input, gate_up_weight, gate_up_scale, bias_arg, problem);
#else
    (void)input;
    (void)gate_up_weight;
    (void)gate_up_scale;
    (void)gate_up_bias;
    TORCH_CHECK(false,
                "cutlass_gated_mlp_staged_producer_sm120_fp8 was not compiled with "
                "CUTLASS_ARCH_MMA_SM120_SUPPORTED. Rebuild with --config=cuda12_9.");
    return std::make_tuple(torch::Tensor(), torch::Tensor());
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
cutlass_gated_mlp_producer_sm120_fp8(torch::Tensor const&                input,
                                     torch::Tensor const&                gate_up_weight,
                                     torch::Tensor const&                gate_up_scale,
                                     std::optional<torch::Tensor> const& gate_up_bias) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto problem  = check_problem(input, gate_up_weight, gate_up_scale);
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    auto bias_arg = normalize_bias(gate_up_bias, input.device(), input.dtype(), problem.n2);
    return run_reference_producer(input, gate_up_weight, gate_up_scale, bias_arg, problem);
#else
    (void)input;
    (void)gate_up_weight;
    (void)gate_up_scale;
    (void)gate_up_bias;
    TORCH_CHECK(false,
                "cutlass_gated_mlp_producer_sm120_fp8 was not compiled with "
                "CUTLASS_ARCH_MMA_SM120_SUPPORTED. Rebuild with --config=cuda12_9.");
    return std::make_tuple(torch::Tensor(), torch::Tensor());
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
cutlass_gated_mlp_semi_tail_emulation_sm120_fp8(torch::Tensor const&                input,
                                                torch::Tensor const&                gate_up_weight,
                                                torch::Tensor const&                gate_up_scale,
                                                std::optional<torch::Tensor> const& gate_up_bias) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    auto problem  = check_problem(input, gate_up_weight, gate_up_scale);
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    auto bias_arg = normalize_bias(gate_up_bias, input.device(), input.dtype(), problem.n2);
    return run_semi_tail_emulation(input, gate_up_weight, gate_up_scale, bias_arg, problem);
#else
    (void)input;
    (void)gate_up_weight;
    (void)gate_up_scale;
    (void)gate_up_bias;
    TORCH_CHECK(false,
                "cutlass_gated_mlp_semi_tail_emulation_sm120_fp8 was not compiled with "
                "CUTLASS_ARCH_MMA_SM120_SUPPORTED. Rebuild with --config=cuda12_9.");
    return std::make_tuple(torch::Tensor(), torch::Tensor());
#endif
}
