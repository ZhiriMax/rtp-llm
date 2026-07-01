# SM120 FP8 Gated MLP Fusion Developer Notes

This note records the staged plan for applying a fused gated MLP idea to the
SM120 FP8 blockwise path. It is intentionally scoped to Qwen-style SiGLU MLPs:

```text
gate_up = x @ [W_gate, W_up]
hidden  = silu(gate_up[:, :I]) * gate_up[:, I:]
out     = hidden @ W_down
```

## Current RTP-LLM Path

For `DenseMLP` with `ActivationType.Swiglu`, the current path is:

```text
x
  -> up_proj                         # FP8 blockwise GEMM, BF16 [M, 2I]
  -> silu_and_mul                    # BF16 [M, I]
  -> down_proj                       # FP8 blockwise GEMM, BF16 [M, H]
```

When `RTP_LLM_DENSE_MLP_FUSION=silu_quant` is enabled, the middle part becomes:

```text
x
  -> up_proj                         # BF16 [M, 2I]
  -> fused silu_and_mul + FP8 quant   # QuantizedActivation [M, I]
  -> down_proj                       # consumes QuantizedActivation
```

This keeps the architecture local:

- `DenseMLP` owns MLP-level execution policy.
- `rtp_llm.models_py.modules.fusion.gated_mlp` owns gated MLP fusion
  helpers and the serving-safe `silu_quant` policy.
- `Linear` backends own generic linear and activation-quant contracts.
- No `Linear` backend owns Qwen-specific `[gate, up]` producer semantics.

The legacy switch `ENABLE_DENSE_SILU_MUL_QUANT_FUSION=1` maps to
`RTP_LLM_DENSE_MLP_FUSION=silu_quant` when the new env is not set.

`RTP_LLM_DENSE_MLP_FUSION` is currently an experimental Python-model-local
environment variable, similar to other dense model fast-path switches. It is
not a top-level server argument. Supported values are:

| Value | Behavior |
| --- | --- |
| `off` | Always use the eager path. |
| `silu_quant` | Use fused `silu_and_mul + FP8 quant` when the backend method exists; otherwise use the eager path. |

If `RTP_LLM_DENSE_MLP_FUSION` is set, it takes precedence over the legacy
`ENABLE_DENSE_SILU_MUL_QUANT_FUSION` switch.

`DenseMLP` reads this mode at module initialization time and stores it on the
layer instance. `silu_quant` is an explicit experimental path: if a backend
advertises `quantize_fused_silu_and_mul` but rejects the runtime dtype, shape,
or group size, that error is propagated instead of silently falling back.

The article-style producer boundary is intentionally not exposed through a
serving environment variable yet. The current developer/benchmark-only path is
named `gate_up_staged_producer`, requires the low-level
`cutlass_gated_mlp_staged_producer_sm120_fp8` op and compatible SM120 FP8
layers, and fails fast if those requirements are not met. The op uses a
staged-parity body that calls the existing input quantization, SM120 gate/up
GEMM, and fused `silu_and_mul + FP8 quant` kernels. It validates the low-level
producer contract, but it still materializes the BF16 `[M, 2I]` intermediate
and is not the final performance kernel. For the gate/up input quantization
step, it follows the same `FP8_GROUP_QUANT_KERNEL` v1/v2 selection as the
Python staged path.

The same Python wrapper can also be exercised manually with the developer-only
`gate_up_producer` mode. That mode calls
`cutlass_gated_mlp_producer_sm120_fp8`, i.e. the current reference/full
producer ABI. It remains excluded from `RTP_LLM_DENSE_MLP_FUSION` serving
values, so enabling it requires an explicit internal test hook rather than an
environment variable.

## What the Article-Style Full Fusion Would Change

The target full fusion is not just the current `silu_quant` path. It should
remove the BF16 `[M, 2I]` global-memory intermediate produced by `up_proj`.

Target path:

```text
x
  -> fused gate/up GEMM + silu_and_mul + dynamic FP8 quant
  -> QuantizedActivation [M, I]
  -> down_proj
```

The fused kernel must produce the exact contract consumed by
`CudaFp8VllmBlockwiseLinear._forward_quantized`:

- data dtype: `torch.float8_e4m3fn`
- scale dtype: `torch.float32`
- group size: 128 columns
- scale shape: `(M, I / 128)`
- scale layout: column-major with token stride 1 and K-group stride `M`
- quant key/layout equal to the SM120 blockwise input quant key

The current developer-only reference producer already uses the future low-level
full-op ABI and directly returns FP8 activation data plus scales, but its
dot-product body is a tiled CUDA reference implementation, not the final
SM120 tensor-MMA/CUTLASS producer:

```text
data, scale = cutlass_gated_mlp_producer_sm120_fp8(
    x,
    gate_up_weight,
    gate_up_weight_scales,
    gate_up_bias,
)
```

`rtp_llm.models_py.modules.fusion.gated_mlp` wraps `(data, scale)` into a
`QuantizedActivation` and lets the existing `down_proj` consume it.
The serving wrapper only admits 2D BF16 input and SM120 FP8-compatible
projection layers where `up_proj.N == 2 * down_proj.K` and `down_proj.K` is
divisible by the 128-column activation quantization group size. Empty `M=0`
batches are not sent into the producer boundary; the normal serving path should
use the fallback path until the low-level scale-stride and downstream
empty-GEMM contracts are tested on device. The explicit developer staged
producer mode instead fails fast for incompatible inputs, because it exists to
validate the producer contract.

For `staged-equivalent`, the full op must also reproduce the current
`up_proj` input quantization semantics before the gate/up GEMM: BF16 input is
quantized to FP8 E4M3 with per-token, per-128-column dynamic scales, matching
the current SM120 blockwise linear input quantization contract. The returned
activation `scale` must match the existing quant kernel's output scale
semantics consumed by down-proj, i.e. `amax / 448` per token group, not the
quantization multiplier `448 / amax`.

## Mapping the Article Kernel to RTP-LLM

The article's speedup comes from a different gate/up GEMM organization, not
from a regular output epilogue attached to the existing `(M, 2I, H)` GEMM.
The important kernel-level properties are:

- Single tensor-MMA over interleaved `up + gate` rows: the tensor pipe computes
  both halves in one MMA issue path instead of two serialized GEMMs. A normal
  `cutlass_scaled_mm_blockwise_sm120_fp8` tile over `(M, 2I, H)` cannot pair
  `j` with `j + I` inside the same output visitor.
- The article's `64 up + 64 gate` organization is a producer/mainloop layout,
  not a serving-visible weight contract. It can be implemented either by
  preprocessing B into that layout, or by using producer-side 3D TMA/load logic
  to gather `[up[j:j+64], gate[j:j+64]]` into contiguous shared memory. In
  both cases, the producer must still handle the original independent gate/up
  block scales correctly. The current generic SM120 GEMM assumes one B scale
  per 128 output rows, so a naive 64-row interleave cannot be represented by
  its existing `B_sf` contract.
- N-split math warpgroups sharing the same A tile: two math WGs consume one
  loaded A tile and different B tiles, improving A reuse and tensor pipe
  occupancy. This is a producer-specific mainloop/scheduler choice, not a
  Python `DenseMLP` or generic linear policy.
- Deferred scale/FMA work across K blocks: scale compensation work is
  overlapped with TMA/mainloop work instead of sitting on the critical path
  after each MMA. The current CUTLASS factory does not expose this as a
  local epilogue knob.
- FP8 output generated in-kernel using amax + STSM/TMA store: this removes
  BF16 `[M, 2I]` global output and a later activation/quant readback. The
  producer must write both FP8 data and FP32 scales in the existing
  `QuantizedActivation` layout.

Therefore the current staged producer is intentionally only an ABI and
correctness boundary. It still calls the generic SM120 blockwise GEMM and still
materializes BF16 `[M, 2I]`, so it cannot show the article's main win.

## Recommended Next Implementation Step

The lowest-risk implementation step beyond the current staged producer is a
"semi-true" producer:

```text
x
  -> input FP8 quant
  -> producer-specific gate/up GEMM writes BF16 activated [M, I]
  -> existing per_token_group_quant_8bit_v2 non-fused quant
  -> QuantizedActivation [M, I]
  -> down_proj
```

This still has a separate activation quant kernel, but it removes the full
BF16 `[M, 2I]` gate/up intermediate. It keeps the amax reduction and FP8
scale-store semantics inside the existing quant kernel while the new CUTLASS
work focuses on making paired gate/up accumulators available before store.
That is closer to the article than the staged producer, but avoids solving
custom paired-accumulator epilogue and group-scale store in one step.

The producer-specific CUTLASS path should be cloned from the SM120 blockwise
GEMM factory rather than modifying the generic linear GEMM. First version
should explicitly avoid the `swap_ab` dispatch path because transposed problem
mapping complicates token/group scale layout and bias direction.

This semi-true step is still below the article's FP8-output kernel. It removes
the full BF16 `[M, 2I]` store/read only after the new gate/up GEMM writes
activated BF16 `[M, I]` directly. It does not yet implement STSM/TMA FP8
stores, cross-WG amax exchange, or the article's N-split producer/consumer
pipeline.

The current C++ producer is structured around this replacement point:

```text
quantize_input(input)
activated = producer_body(input_fp8, input_scales, gate_up_weight, gate_up_scale)
quantize_activated_tail(activated)
```

The semi-true producer should keep `quantize_input` and
`quantize_activated_tail` unchanged, and replace only the producer body with a
producer-specific GEMM that writes activated BF16 `[M, I]`. The Python ABI
remains `(data, scale)`.

The current `cutlass_gated_mlp_producer_sm120_fp8` goes one step further for
the reference path:

```text
quantize_input(input)
producer_body writes FP8 data and FP32 scale directly
```

That locks the full FP8-output ABI before the SM120 tensor-MMA body exists. The remaining
performance work is to replace the tiled CUDA dot-product body with an article-style
producer/mainloop that computes paired gate/up accumulators efficiently.
In code, this replacement is intentionally isolated behind
`run_full_output_producer(FullOutputProducerInputs)`: Python, pybind, input
quantization, output `(data, scale)` ownership, and down-proj consumption should
stay unchanged while the tiled reference producer body is replaced.
Inside the CUDA reference body, the intended replacement is even narrower:
`load_a_tile`, `load_paired_b_tile`, and `store_quantized_activation` preserve
the current ABI and data layout, while the producer body type fills
`PairedAccumulator<Tile>`. The current launcher is deliberately named
`compute_cuda_reference_full_output_producer<Body>` because it uses the scalar
CUDA reference launch shape. The body owns its `TileShape` trait, but a future
SM120 tensor-MMA implementation should introduce its own SM120 tensor-MMA/CUTLASS launcher instead
of pretending the 1024-thread reference launcher is generic. The current body
is `CudaReferencePairedBody<CudaReferenceProducerTile>` and uses scalar dot
products (`k_uses_tensor_mma=false`). The final SM120 tensor-MMA producer should provide
another body or launcher with the same public gate/up accumulator contract
before `store_quantized_activation`. The reference launcher has a compile-time
guard against `k_uses_tensor_mma=true` bodies to make that boundary explicit.
`PairedAccumulator<Tile>` is a per-thread fragment: internally it stores each
lane's local `up + gate` pair while the CTA-level 128-column grouping remains
represented by the paired shared B tile.

The current `cutlass_gated_mlp_semi_tail_emulation_sm120_fp8` op validates the
tail half of that plan only:

```text
input FP8 quant
  -> staged ordinary gate/up GEMM still writes BF16 [M, 2I]
  -> small activation kernel writes BF16 activated [M, I]
  -> existing non-fused quant writes QuantizedActivation [M, I]
```

It exists to measure the extra BF16-activated rounding and non-fused quant tail
cost. It is not the semi-true producer because the gate/up GEMM is still the
ordinary staged GEMM. The non-fused tail follows the same
`FP8_GROUP_QUANT_KERNEL` v1/v2 selection as Python `_quantize_input`.

`cutlass_gated_mlp_producer_sm120_fp8` is the current full-output reference
producer:

```text
input FP8 quant
  -> producer-specific CUDA kernel directly writes FP8 activated [M, I]
     plus FP32 dynamic scales
```

This op removes the staged BF16 `[M, 2I]` materialization, so it is a real
dataflow step beyond staged parity. It is still not the final article kernel:
the gate/up dot products are computed by a tiled CUDA reference kernel rather
than a SM120 tensor-MMA/CUTLASS producer-consumer mainloop.

`ArticleMmaProducerTarget` records the intended performance contract from the
article:

- MMA tile shape: 64 tokens by 128 output columns by 128 K columns.
- Instruction tile: `m64n128k32`, so one 128-K tile has four MMA steps.
- The 128 output columns are paired as `64 up + 64 gate`, so one tensor-MMA issue
  path produces both operands needed by SiLU-mul.
- CTA schedule: `2 math warpgroups + 1 TMA/producer warpgroup`, i.e.
  `384` threads. The two math warpgroups own different 64-column output slices
  and share the same A tile.
- The final SM120 tensor-MMA body should keep gate/up accumulators in registers, apply
  independent gate/up scale compensation during or around the mainloop, then
  run activation, amax, quantization, and stores without writing BF16
  `[M, 2I]`.

The code keeps these constants in `ArticleMmaProducerTile`: `k_mma_k=32`,
`k_mma_steps=4`, `k_acc_registers=64`, and `k_cta_threads=384`. They are not
used by the scalar CUDA reference launcher, but they make the intended tensor-MMA
body contract explicit and reviewable.

`ArticleMmaPairedBody` is the corresponding body contract placeholder:
it owns `ArticleMmaProducerTile` and marks `k_uses_tensor_mma=true`. It has no
`run()` body yet. This is intentional: the scalar reference launcher has a
compile-time guard against tensor-MMA bodies, so the future implementation must
add a SM120/CUTLASS-specific launcher instead of accidentally running under the
validation-only CUDA reference launch shape.

`CudaReferenceProducerTile` deliberately is not that schedule. It uses the same
64-token by 128-column by 128-K data grouping and paired B layout, but maps
each output lane to scalar CUDA threads instead of tensor-core MMA fragments.
It keeps the 128-wide A tile as FP8 in shared memory, and arranges B as two
64-column pairs where each pair is `up[0:64] + gate[0:64]` in shared memory.
The body computes raw FP8 dot products and applies input/weight scale
compensation for each K tile before adding into paired gate/up accumulators.
It then directly writes FP8 activation data plus scales. This mirrors the
article's "load A once, feed paired gate/up outputs" dataflow and the
`64 up + 64 gate` producer layout without yet using tensor-core MMA. It
exists to lock down the ABI, dynamic-scale layout, rounding target, and
benchmark harness before replacing the dot-product body with the performance
kernel.

The scale contract is intentionally separated from the shared-memory B order:
`load_paired_b_tile` stages B as `up + gate`, while
`load_paired_scale_coeff` still reads the original weight-scale layout as
`gate groups` followed by `up groups`. Future producer bodies should reuse that
semantic boundary instead of deriving scale offsets ad hoc.

The CUDA reference CTA uses `64 * 16 = 1024` threads and about 45KB of shared
memory: 8KB for FP8 A, 32KB for paired FP8 B, and 4KB for row-local amax. This
is intentionally a validation shape. The expected high-performance scheduling
shape is the `384`-thread warp-specialized target above, not the reference CTA.

The intended code cut is deliberately narrow:

- keep `cutlass_scaled_mm_blockwise_sm120_fp8` unchanged;
- add a producer-specific launcher beside the staged producer;
- keep `load_a_tile`, `load_paired_b_tile`, and `store_quantized_activation`
  as the stable reference dataflow while replacing only the producer body type
  that fills `PairedAccumulator<Tile>`;
- keep A/B staging in FP8 and make scale compensation part of the producer body
  contract, because a future tensor-MMA body will also accumulate raw FP8 fragments
  before applying per-token/per-block scales;
- clone only the scale/layout/problem-shape validation needed from the generic
  SM120 blockwise GEMM;
- start with the non-`swap_ab` shape mapping only;
- make unsupported schedules fail fast in the low-level producer instead of
  silently falling back to the generic GEMM.

Doing another ordinary `(M, 2I, H)` GEMM inside this boundary is not considered
progress toward the semi-true producer; it is exactly the staged baseline.

## Correctness Contract

The first correctness target should match the current RTP-LLM staged path:

```text
gate_up_bf16 = up_proj(x)
quantized    = down_proj.quantize_fused_silu_and_mul(gate_up_bf16)
out          = down_proj(quantized)
```

The full kernel can be made equivalent only if the selected implementation
target also matches the rounding and scale contract below:

- The activation type is `Swiglu`.
- The gate/up order is `[gate, up]`.
- The pre-merged `W.ffn_w13` path preserves the same gate/up order.
- `H` and `I` are divisible by the SM120 input quant group size, currently
  128. The staged producer rejects unaligned shapes instead of padding.
- Gate/up bias, if present, is added before `silu(gate) * up`.
- The dynamic output scale is computed from `silu(gate) * up`, not inferred
  from input or weight scales.
- The output amax is reduced per token and per 128 intermediate columns.
- Output scale stores the dequant scale `amax / 448`, with `eps=1e-4`,
  FP8 E4M3 range `[-448, 448]`, and `scale_ue8m0=false`.
- The output `QuantizedActivation` layout matches the down projection consumer.
- The FP8 activation data and down projection output are contiguous tensors.
- `FP8_GROUP_QUANT_KERNEL` should be fixed before process/module
  initialization. Python caches the selected group-quant kernel at import time,
  while the staged C++ op reads the same environment for its internal helper.

There are two valid implementation targets:

- `staged-equivalent`: reproduce the current staged rounding points. Gate/up
  GEMM accumulators are rounded to BF16 after bias, then `silu(gate)` is
  rounded to BF16 before multiplying by BF16 `up`, and amax/FP8 quantization is
  computed from that low-precision product. This is the right target for
  bitwise-oriented validation against the current path.
- `tolerance-equivalent`: apply activation directly from FP32 accumulators and
  compute amax/FP8 quantization from that product. This may be faster and
  closer to the article's kernel philosophy, but it is not bitwise-equivalent
  to the current staged path. Validate with numerical tolerances aligned with
  existing FP8 activation-quant tests.

## Why the Existing CUTLASS GEMM Cannot Be Reused Directly

The current SM120 blockwise CUTLASS op:

```text
cutlass_scaled_mm_blockwise_sm120_fp8(D, A, B, A_sf, B_sf, bias)
```

computes one GEMM and has a regular BF16/FP16 output epilogue. Full gated MLP
fusion needs extra behavior that the current epilogue does not provide:

- access to paired gate/up accumulators;
- `silu(gate) * up` inside the epilogue or mainloop-adjacent path;
- per-token, per-128-column amax reduction over the activated output;
- FP8 output plus dynamic scale output in the exact down-proj input layout.

This is a new kernel boundary, not a tile-dispatch tweak.

In code, the generic SM120 path builds:

```text
cutlass_3x_gemm_fp8_blockwise
  -> Sm120BlockwiseScaleConfig<...>
  -> LayoutSFA/LayoutSFB
  -> CollectiveBuilder mainloop
  -> LinCombPer{Row,Col}Bias output operation
```

That structure is correct for one logical output matrix `D = A @ B^T`, where
each output element has one accumulator and one output location. The gated
producer needs a different mainloop-visible B layout: the math body must see
`64 up + 64 gate` for the same logical output group, while still loading the
original gate/up scales independently. That pairing is not representable by
only changing `MmaTileShape`, `KernelSchedule`, or the existing
`LinCombPerColBias` epilogue.

The local vLLM/SGLang C3x EVT references are useful for fusing elementwise
scale, bias, and zero-point corrections, but they are still output-element
visitor trees. The gated MLP producer needs information that is not naturally
available to a regular single-output element visitor:

- pair the accumulator for output column `j` with the accumulator for
  `j + I`;
- compute `silu(gate) * up` before the output write;
- reduce amax over each `(token, 128-column group)`;
- write two logical outputs, FP8 data and FP32 scale.

So the low-intrusion path is not "swap the current `DefaultOperation` with a
larger EVT". The implementation should either use a custom CUTLASS kernel
body/collective that owns the paired gate/up output tile and scale store, or a
hand-written SM120 tensor-MMA producer/consumer kernel following the article's structure.

The concrete blocker is the current GEMM problem shape `(M, 2I, H)`: a normal
128-column epilogue tile sees either a gate tile or an up tile, while the paired
value lives `I` columns away. A regular output visitor cannot combine those two
remote accumulator tiles. In addition, the FP8 output contract requires a
group reduction for each `(token, 128 output columns)` and a second FP32 scale
store. Both requirements exceed the current `LinCombPerColBias`-style single
matrix epilogue.

## Existing Tensor-MMA Building Blocks

The H100 article is written around Hopper WGMMA/QGMMA terminology. That does
not map directly to the SM120 path in this repo. The current SM120 blockwise
GEMM uses CUTLASS 4/CUTE with `cute::UMMA` scale layout concepts and
SM120-specific schedules:

```text
cutlass_scaled_mm_blockwise_sm120_fp8_kernels.cu
  -> cutlass_3x_gemm_fp8_blockwise
  -> cute::UMMA::Major::{MN,K}
  -> cutlass::arch::Sm120
  -> KernelTmaWarpSpecializedBlockwise{Pingpong,Cooperative}Sm120
```

The repository also has Hopper-era QGMMA wrappers in FMHA/XQA, but those are
attention-specific and guarded around SM90/Hopper assumptions. They should not
be pulled into the SM120 dense MLP path. The gated MLP producer still needs its
own bridge from the blockwise GEMM tensors into a SM120 collective-compatible
layout:

- construct the producer's staged A tile and paired B tile layout;
- map the MMA accumulator fragment back to paired `up/gate` accumulator
  lanes before activation;
- preserve separate gate/up weight scales even though the shared-memory B tile
  is presented as paired `64 up + 64 gate`;
- coordinate the 128-column amax reduction and FP8 data/scale stores.

So the next performance step is not to call an existing attention kernel. It is
to add a new `PairedAccumulator`-compatible SM120/CUTLASS collective or
producer launcher under the current producer boundary.

The existing SM120 blockwise GEMM path in this directory already uses the
CUTLASS/CUTE collective stack:

```text
cutlass_scaled_mm_blockwise_sm120_fp8_kernels.cu
  -> cutlass_3x_gemm_fp8_blockwise
  -> CollectiveBuilder<Sm120, OpClassTensorOp, ...>
  -> KernelTmaWarpSpecializedBlockwise{Pingpong,Cooperative}Sm120
```

That is the preferred long-term integration direction for a production
producer body. Pulling the FMHA/XQA Hopper QGMMA wrappers directly into the MLP path
would couple a dense MLP kernel to attention-specific descriptor and
shared-memory machinery. A cleaner implementation should either:

- build a producer-specific CUTLASS collective/epilogue that exposes paired
  gate/up accumulator fragments before store; or
- isolate any future low-level SM120 MMA helper behind a tiny MLP-local adapter
  that owns descriptor construction, fragment mapping, and scale compensation.

In both cases, the public Python/pybind contract remains the current
`cutlass_gated_mlp_producer_sm120_fp8(input, gate_up_weight, scale, bias)`
returning `(data, scale)`.

## Microbenchmark Entry

Use the staged microbench to quantify the current partial fusion and the
remaining full-fusion opportunity:

```bash
/opt/conda310/bin/python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-mlp-stress \
  --m-list 768,1024,1536,2048 \
  --duration 30 \
  --mode both
```

The command reports:

- `baseline`: BF16 `gate_up`, BF16 activation, then down GEMM;
- `fused`: BF16 `gate_up`, fused activation+FP8 quant, then down GEMM;
- BF16 intermediate sizes for `gate_up` and activated hidden.

The `gate_up` intermediate size is the global-memory traffic that a true fused
gated GEMM kernel should remove.

Use the breakdown command to estimate where the staged path spends time:

```bash
/opt/conda310/bin/python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-mlp-breakdown \
  --m-list 768,1024,1536,2048 \
  --warmup 100 \
  --iters 1000
```

The breakdown reports:

- `gate_up_project`: the current gate/up projection path, including input
  quantization and CUTLASS GEMM;
- `silu_mul_quant`: fused activation and FP8 quantization from BF16 `gate_up`;
- `down_quantized_gemm`: down projection consuming the quantized activation;
- `silu_quant_path`: the staged end-to-end partial-fusion path.

`relative_to_silu_quant_path_pct` is a directional attribution aid rather than
an additive profile: each stage is timed independently with selected
intermediates precomputed, while `silu_quant_path` measures the staged path as
one callable.

Use the producer comparison command to verify the low-level C++ producer
boundaries:

```bash
/opt/conda310/bin/python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-producer-compare \
  --m-list 768,1024,1536,2048 \
  --warmup 100 \
  --iters 1000
```

`gated-staged-producer-compare` is kept as a backward-compatible alias for the
same command.

The `staged_producer` row uses `cutlass_gated_mlp_staged_producer_sm120_fp8`.
In the current staged-parity implementation it should be close to `silu_quant`
numerically, but it is not expected to be faster because it still emits the
BF16 `gate_up` intermediate internally.

The `semi_tail_emulation` row uses
`cutlass_gated_mlp_semi_tail_emulation_sm120_fp8`. It still uses the same
ordinary gate/up GEMM, but routes the tail through BF16 activated `[M, I]` plus
non-fused quant. That row estimates the cost and numerical drift of the
planned semi-true producer tail.

The `scale128_interleaved_staged` row is a bench-only layout check. It
reorders the ordinary gate/up weight into scale-compatible 128-row chunks
`[gate128, up128, ...]`, runs the generic GEMM, restores the output to
`[gate, up]`, then uses the same activated-tail reference. This verifies that
weight and scale reordering can preserve semantics under the current 128-row
blockwise scale contract. It is not the article's final `64 up + 64 gate`
MMA tile. The real article layout must live inside a producer-specific
mainloop/TMA path that can load separate gate/up scales while presenting a
paired 128-row tile to tensor-MMA.

The `reference_producer` row uses `cutlass_gated_mlp_producer_sm120_fp8`.
It avoids the BF16 `[M, 2I]` intermediate and the separate non-fused quant tail,
but uses a producer-specific tiled CUDA reference kernel, so it is for
correctness/dataflow comparison rather than performance validation. Its first
correctness target is the activated-tail path, so `diff_vs_tail_*` is more
relevant than `diff_vs_silu_*` for this row.

The comparison prints both `diff_vs_silu_*` and `diff_vs_tail_*`.
`diff_vs_silu_*` measures drift from today's fused `silu_quant` path.
`diff_vs_tail_*` measures the C++ emulation against the Python
`activated BF16 -> non-fused quant` reference, which is the relevant tail
contract for the semi-true producer milestone.

The tiled CUDA `reference_producer` is excluded from the default path list
because it computes gate/up dot products without tensor-core MMA. Use a small
`M` when explicitly validating that dataflow:

```bash
/opt/conda310/bin/python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-producer-compare \
  --m-list 1,2,4 \
  --paths reference_producer,activated_tail_ref \
  --warmup 10 \
  --iters 20
```

Use the opportunity command to estimate the upper bound left for a true fused
producer under the same MLP shape:

```bash
/opt/conda310/bin/python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-fusion-opportunity \
  --m-list 768,1024,1536,2048 \
  --warmup 100 \
  --iters 1000
```

`semi_true_removable_stage_ms` is intentionally conservative: it counts the
standalone `silu_mul_quant` stage that a BF16-activated semi-true producer
should absorb, but it does not pretend that the gate/up GEMM math disappears.
Any extra gain from avoiding BF16 `[M, 2I]` stores and reads must be confirmed
by the real kernel.

For the semi-true producer plan, compare `silu_mul_quant_ms` with
`activated_quant_ms`. The latter estimates the tail cost if the producer writes
BF16 activated `[M, I]` and the existing non-fused quant kernel handles FP8
data/scale generation. `full_fp8_tail_extra_ms` is the additional quant tail
cost relative to today's fused `silu_mul_quant` kernel.

For a true article-style FP8 producer, `article_full_removable_stage_ms`
optimistically counts the larger of those two tail stages because the producer
would do activation, amax, FP8 quant, STSM, and scale store in-kernel.
`article_upper_bound_speedup` is an upper bound for prioritization, not a
promised speedup.

For a true article-style kernel, the gate/up GEMM compute is not removed, but
its output path changes from BF16 `[M, 2I]` global stores to direct activated
FP8 output. The `silu_mul_quant` stage and the BF16 `gate_up` readback are the
most direct kernel-level gap between the current staged path and full fusion.

## Implementation Checklist for the Full Kernel

1. Add a new CUDA/CUTLASS op under the SM120 FP8 blockwise kernel directory.
2. Keep the pybind op low-level: no fallback policy, no model-specific routing.
3. Add the Python wrapper to `rtp_llm.models_py.modules.fusion.gated_mlp`, not
   to a generic `Linear` producer backend.
4. Return a `QuantizedActivation` that the existing down projection consumes.
5. Support Qwen2/Qwen3 gate/up order and bias semantics, or fallback cleanly.
6. Add unit coverage for layout, dtype, shape, scale stride, and numerical diff.
7. Compare end-to-end MLP output against the staged `silu_quant` path before
   comparing service-level performance.

Stable contracts already shared with the staged producer:

- `input` is BF16 `(M, H)`.
- input activation quantization uses FP8 E4M3 with per-token, per-128-column
  scales laid out as `(M, H / 128)` with stride `(1, M)`.
- input activation quantization uses `eps=1e-4`, FP8 range `[-448, 448]`,
  and dequant-scale output semantics `amax / 448`.
- `gate_up_weight` is FP8 E4M3 `(2I, H)` row-major contiguous.
- `gate_up_scale` is FP32 `(ceil_div(2I, 128), ceil_div(H, 128))` with
  K-major stride `(ceil_div(H, 128), 1)`.
- if `gate_up_bias` exists, it is BF16 on the same device, has `2I`
  elements, and is applied independently to gate/up before activation.
- output `data` is FP8 E4M3 `(M, I)` contiguous.
- output `scale` is FP32 `(M, I / 128)` with stride `(1, M)`.
- output `scale` uses dequant-scale semantics `amax / 448`, not the quant
  multiplier `448 / amax`.
- gate/up order is `[gate, up]`.

Unstable implementation choices for the true kernel:

- whether to target staged-equivalent BF16 rounding or article-style
  tolerance-equivalent FP32 accumulator activation;
- how to organize paired gate/up output tiles so `j` and `j + I` are available
  before store;
- where to perform the per-token 128-column amax reduction;
- whether scale is written directly from the GEMM kernel or through a small
  follow-up reduction kernel during early development.

Minimal source touch points for the CUDA path:

- C++/CUDA kernel and header:
  `rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/`
  `fp8_blockwise_sm120/cutlass_gated_mlp_producer_sm120_fp8.{h,cu}`
- CUDA BUILD target:
  `rtp_llm/models_py/bindings/cuda/cutlass/BUILD`
- pybind registration:
  `rtp_llm/models_py/bindings/cuda/RegisterCudaOps.cc`
- Python op import surface:
  `rtp_llm.ops.compute_ops`
- MLP-level Python wrapper:
  `rtp_llm/models_py/modules/fusion/gated_mlp.py`
- Execution policy:
  `rtp_llm/models_py/modules/hybrid/dense_mlp.py`
