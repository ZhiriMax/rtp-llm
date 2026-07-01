#pragma once

#include <optional>
#include <tuple>

#include <torch/all.h>

// Experimental gated MLP producer boundaries for SM120 FP8.
//
// The staged producer proves the low-level ABI by reusing existing input quant,
// gate/up GEMM, and fused silu/mul quant kernels. The reference producer below
// proves the final FP8-output dataflow by avoiding the BF16 [M, 2I]
// intermediate and directly writing activation FP8 data plus scales, but it is
// still a tiled CUDA reference kernel. The final article-style producer should
// replace that dot-product body with a producer-specific tensor-MMA or CUTLASS
// kernel that owns paired gate/up accumulators before store; this is not a
// generic linear epilogue extension.
//
// Planned contract:
//   input:          (M, H) bf16 activation
//   gate_up_weight: (2I, H) float8_e4m3fn row-major contiguous weight
//   gate_up_scale:  blockwise FP32 scales for gate_up_weight
//   gate_up_bias:   optional (2I,) bf16 bias
//
// Output:
//   data:  (M, I) float8_e4m3fn, contiguous
//   scale: (M, I / 128) float32, column-major scale layout stride (1, M)
//
// The Python wrapper turns (data, scale) into QuantizedActivation and feeds
// existing down_proj unchanged.
std::tuple<torch::Tensor, torch::Tensor>
cutlass_gated_mlp_staged_producer_sm120_fp8(torch::Tensor const&                input,
                                            torch::Tensor const&                gate_up_weight,
                                            torch::Tensor const&                gate_up_scale,
                                            std::optional<torch::Tensor> const& gate_up_bias = std::nullopt);

// Experimental full-output reference producer. It keeps the existing input
// quantization contract, but replaces the staged BF16 [M, 2I] GEMM output and
// separate activation-quant tail with a producer-specific CUDA kernel that
// directly writes FP8 activated [M, I] plus dynamic scales. The reference body
// computes a 64-token by 128-column CUDA-reference output tile, stages FP8 A
// and per-64-column `up + gate` FP8 B pairs in shared memory, applies
// input/weight scale compensation inside the body, and directly writes data
// plus scales. It is still a correctness/dataflow milestone, not the final
// SM120 tensor-MMA/CUTLASS performance kernel. The reference CTA mirrors the article's
// 64-token by 128-column grouping, but not its 2-math-WG + 1-producer-WG
// schedule; internally, the body type fills a paired gate/up accumulator while
// load/store and the public op contract stay fixed.
std::tuple<torch::Tensor, torch::Tensor>
cutlass_gated_mlp_producer_sm120_fp8(torch::Tensor const&                input,
                                     torch::Tensor const&                gate_up_weight,
                                     torch::Tensor const&                gate_up_scale,
                                     std::optional<torch::Tensor> const& gate_up_bias = std::nullopt);

// Developer-only tail emulation for the semi-true producer plan. It still uses
// the staged gate/up GEMM, then writes BF16 activated [M, I] and quantizes that
// with the existing non-fused activation quant kernel.
std::tuple<torch::Tensor, torch::Tensor>
cutlass_gated_mlp_semi_tail_emulation_sm120_fp8(torch::Tensor const&                input,
                                                torch::Tensor const&                gate_up_weight,
                                                torch::Tensor const&                gate_up_scale,
                                                std::optional<torch::Tensor> const& gate_up_bias = std::nullopt);
