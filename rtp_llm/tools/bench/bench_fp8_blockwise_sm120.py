"""SM120 FP8 blockwise GEMM operator microbenchmarks.

Examples:
  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 bare-gemm \
      --case all --m-list 768,1024,1536,2048

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 compare-bare \
      --case all --m-list 768,1024,1536,2048 --configs legacy,auto

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 compare-bare \
      --custom-shapes 1024x896x1152,1024x4864x896 --configs legacy,auto

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 rms-stress \
      --case qkv --m 1024 --duration 30 --mode both

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 silu-stress \
      --m 1536 --duration 30 --mode both

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-mlp-stress \
      --m-list 1024,1536 --duration 30 --mode both

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-mlp-breakdown \
      --m-list 1024,1536

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-producer-compare \
      --m-list 1024,1536

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 gated-fusion-opportunity \
      --m-list 1024,1536
"""

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.modules.fusion.fp8_sm120_contract import (
    SM120_FP8_INPUT_GROUP_SIZE,
    make_sm120_fp8_activation,
)


def _optional_compute_op(name: str) -> Optional[Callable]:
    try:
        import rtp_llm.ops.compute_ops as compute_ops
    except ImportError:
        return None
    return getattr(compute_ops, name, None)


cutlass_scaled_mm_blockwise_sm120_fp8 = _optional_compute_op(
    "cutlass_scaled_mm_blockwise_sm120_fp8"
)
cutlass_gated_mlp_producer_sm120_fp8 = _optional_compute_op(
    "cutlass_gated_mlp_producer_sm120_fp8"
)
cutlass_gated_mlp_semi_tail_emulation_sm120_fp8 = _optional_compute_op(
    "cutlass_gated_mlp_semi_tail_emulation_sm120_fp8"
)
cutlass_gated_mlp_staged_producer_sm120_fp8 = _optional_compute_op(
    "cutlass_gated_mlp_staged_producer_sm120_fp8"
)


QWEN2_05B_HIDDEN = 896
QWEN2_05B_INTERMEDIATE = 4864
QWEN2_05B_QKV = 1152
GATED_PRODUCER_PATHS = (
    "baseline",
    "silu_quant",
    "activated_tail_ref",
    "scale128_interleaved_staged",
    "staged_producer",
    "semi_tail_emulation",
    "reference_producer",
)
DEFAULT_GATED_PRODUCER_PATHS = (
    "baseline",
    "silu_quant",
    "activated_tail_ref",
    "scale128_interleaved_staged",
    "staged_producer",
    "semi_tail_emulation",
)


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    k: int
    n: int


@dataclass(frozen=True)
class GemmBenchCase:
    name: str
    m: int
    k: int
    n: int


PROJECTION_SHAPES: Dict[str, ProjectionShape] = {
    "qkv": ProjectionShape("qkv", QWEN2_05B_HIDDEN, QWEN2_05B_QKV),
    "o_proj": ProjectionShape("o_proj", QWEN2_05B_HIDDEN, QWEN2_05B_HIDDEN),
    "gate_up": ProjectionShape(
        "gate_up", QWEN2_05B_HIDDEN, QWEN2_05B_INTERMEDIATE * 2
    ),
    "down": ProjectionShape("down", QWEN2_05B_INTERMEDIATE, QWEN2_05B_HIDDEN),
}


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SM120 FP8 microbench")
    if cutlass_scaled_mm_blockwise_sm120_fp8 is None:
        raise RuntimeError("cutlass_scaled_mm_blockwise_sm120_fp8 is not available")


def _print_header(title: str) -> None:
    _require_cuda()
    print(torch.cuda.get_device_name())
    print(title)
    env_keys = ["FP8_BLOCKWISE_SM120_FORCE_CONFIG"]
    active_env = {key: os.environ.get(key) for key in env_keys if os.environ.get(key)}
    if active_env:
        print("dispatch env: " + ", ".join(f"{k}={v}" for k, v in active_env.items()))


def _parse_m_list(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--m-list must contain at least one value")
    return values


def _parse_custom_shapes(raw: Optional[str]) -> List[GemmBenchCase]:
    if not raw:
        return []
    cases = []
    for idx, item in enumerate(raw.split(",")):
        item = item.strip().lower()
        if not item:
            continue
        dims = item.replace("*", "x").split("x")
        if len(dims) != 3:
            raise ValueError(
                f"invalid custom shape {item!r}, expected MxKxN, e.g. 1024x896x1152"
            )
        m, k, n = (int(dim) for dim in dims)
        cases.append(GemmBenchCase(f"custom{idx}", m, k, n))
    return cases


def _parse_path_list(raw: str, choices: Tuple[str, ...]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(choices)
    aliases = {"interleaved_staged": "scale128_interleaved_staged"}
    values = [aliases.get(item.strip(), item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in choices]
    if unknown:
        raise ValueError(f"unknown path(s) {unknown}, expected all or one of {choices}")
    if not values:
        raise ValueError("--paths must contain at least one path")
    return values


def _select_cases(case: str) -> Iterable[ProjectionShape]:
    if case == "all":
        return PROJECTION_SHAPES.values()
    if case not in PROJECTION_SHAPES:
        raise ValueError(f"unknown case={case}, expected one of all/{list(PROJECTION_SHAPES)}")
    return [PROJECTION_SHAPES[case]]


def _iter_bare_cases(
    case: str, m_list: str, custom_shapes: Optional[str]
) -> Iterable[GemmBenchCase]:
    custom = _parse_custom_shapes(custom_shapes)
    if custom:
        return custom
    m_values = _parse_m_list(m_list)
    return [
        GemmBenchCase(shape.name, m, shape.k, shape.n)
        for shape in _select_cases(case)
        for m in m_values
    ]


def _make_blockwise_linear(k: int, n: int) -> CudaFp8VllmBlockwiseLinear:
    weight = torch.empty(k, n, device="cuda", dtype=torch.float8_e4m3fn)
    weight_scales = torch.rand(
        (k + 127) // 128,
        (n + 127) // 128,
        device="cuda",
        dtype=torch.float32,
    )
    return CudaFp8VllmBlockwiseLinear(
        weight=weight,
        weight_scales=weight_scales,
        input_scales=None,
        bias=None,
        quant_config=None,
    )


def _make_bare_gemm_case(case: GemmBenchCase) -> Tuple[Callable[[], torch.Tensor], int]:
    linear = _make_blockwise_linear(case.k, case.n)
    x = torch.randn(case.m, case.k, device="cuda", dtype=torch.bfloat16)
    input_fp8, input_scales = linear._quantize_input(x)
    out = torch.empty(case.m, case.n, device="cuda", dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        cutlass_scaled_mm_blockwise_sm120_fp8(
            out,
            input_fp8,
            linear.weight,
            input_scales,
            linear.weight_scales,
            None,
        )
        return out

    flops = 2 * case.m * case.k * case.n
    return run, flops


def _scale128_interleaved_gate_up_indices(
    intermediate_size: int, block: int = SM120_FP8_INPUT_GROUP_SIZE
) -> torch.Tensor:
    """Return a staged-GEMM-only interleave that preserves 128-row scale groups.

    The article's real producer wants 64 up rows followed by 64 gate rows in
    shared memory so one tensor-MMA can produce a paired tile. The generic
    `cutlass_scaled_mm_blockwise_sm120_fp8` contract, however, has one B scale
    per 128 output rows. This bench helper therefore uses 128-row chunks only:
    it checks that weight/scale permutation is semantically safe under today's
    generic GEMM contract, not that the final 64+64 producer exists.
    """
    if intermediate_size % block != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by {block}"
        )
    chunks = []
    for start in range(0, intermediate_size, block):
        chunks.extend(range(start, start + block))
        chunks.extend(
            range(intermediate_size + start, intermediate_size + start + block)
        )
    return torch.tensor(chunks, device="cuda", dtype=torch.long)


def _restore_gate_up_from_scale128_interleaved(
    gate_up: torch.Tensor, block: int = SM120_FP8_INPUT_GROUP_SIZE
) -> torch.Tensor:
    intermediate_size = gate_up.shape[-1] // 2
    if intermediate_size % block != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by {block}"
        )
    restored = torch.empty_like(gate_up)
    for chunk_id, start in enumerate(range(0, intermediate_size, block)):
        src = chunk_id * block * 2
        restored[:, start : start + block] = gate_up[:, src : src + block]
        restored[:, intermediate_size + start : intermediate_size + start + block] = (
            gate_up[:, src + block : src + block * 2]
        )
    return restored


def _make_scale128_interleaved_gate_up(
    gate_up: CudaFp8VllmBlockwiseLinear,
) -> CudaFp8VllmBlockwiseLinear:
    intermediate_size = gate_up.N // 2
    index = _scale128_interleaved_gate_up_indices(intermediate_size)
    scale_index = _scale128_interleaved_gate_up_indices(
        (gate_up.N + SM120_FP8_INPUT_GROUP_SIZE - 1) // SM120_FP8_INPUT_GROUP_SIZE,
        block=1,
    )
    weight = gate_up.weight.index_select(0, index).contiguous()
    weight_scales = gate_up.weight_scales.index_select(0, scale_index).contiguous()
    bias = None
    if gate_up.bias is not None:
        bias = gate_up.bias.reshape(-1).index_select(0, index).contiguous()
    return CudaFp8VllmBlockwiseLinear(
        weight=weight.reshape(gate_up.K, gate_up.N),
        weight_scales=weight_scales.reshape(gate_up.scale_K, gate_up.scale_N),
        input_scales=None,
        bias=bias,
        quant_config=None,
    )


def _measure_ms(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    mean = sum(times) / len(times)
    p50 = times[len(times) // 2]
    return mean, p50, times[0]


def _run_stress(
    name: str,
    fn: Callable[[], torch.Tensor],
    duration_s: float,
    report_interval_s: float,
) -> Tuple[int, float, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    next_report = start + report_interval_s
    last_report = start
    last_iters = 0
    iters = 0
    print(f"\ncase={name} duration={duration_s:.1f}s")
    while True:
        fn()
        iters += 1
        now = time.perf_counter()
        if now - start >= duration_s:
            break
        if now >= next_report:
            torch.cuda.synchronize()
            sync_now = time.perf_counter()
            window_iters = iters - last_iters
            window_s = sync_now - last_report
            ops_s = window_iters / window_s
            print(
                f"  t={sync_now - start:7.2f}s iters={iters:8d} "
                f"window_ops/s={ops_s:10.2f} avg_ms/op={1000.0 / ops_s:8.4f}"
            )
            last_report = sync_now
            last_iters = iters
            next_report = sync_now + report_interval_s
    torch.cuda.synchronize()
    total_s = time.perf_counter() - start
    ops_s = iters / total_s
    avg_ms = 1000.0 / ops_s
    print(
        f"done case={name} total_iters={iters} total_s={total_s:.3f} "
        f"ops/s={ops_s:.2f} avg_ms/op={avg_ms:.4f}"
    )
    return iters, ops_s, avg_ms


def bare_gemm(args: argparse.Namespace) -> None:
    _print_header("Running bare cutlass_scaled_mm_blockwise_sm120_fp8 microbench...")
    for bench_case in _iter_bare_cases(args.case, args.m_list, args.custom_shapes):
        run, flops = _make_bare_gemm_case(bench_case)
        mean, p50, min_ms = _measure_ms(run, args.warmup, args.iters)
        tflops = flops / (mean / 1000.0) / 1e12
        print(
            f"{bench_case.name:<8} M={bench_case.m:5d} K={bench_case.k:5d} N={bench_case.n:5d} "
            f"mean={mean:8.4f} ms p50={p50:8.4f} ms min={min_ms:8.4f} ms "
            f"{tflops:8.2f} TFLOPS"
        )


def _set_force_config(config: str) -> Optional[str]:
    old_value = os.environ.get("FP8_BLOCKWISE_SM120_FORCE_CONFIG")
    os.environ["FP8_BLOCKWISE_SM120_FORCE_CONFIG"] = config
    return old_value


def _restore_force_config(old_value: Optional[str]) -> None:
    if old_value is None:
        os.environ.pop("FP8_BLOCKWISE_SM120_FORCE_CONFIG", None)
    else:
        os.environ["FP8_BLOCKWISE_SM120_FORCE_CONFIG"] = old_value


def compare_bare(args: argparse.Namespace) -> None:
    _print_header("Comparing bare cutlass_scaled_mm_blockwise_sm120_fp8 configs...")
    configs = [config.strip() for config in args.configs.split(",") if config.strip()]
    bench_cases = list(_iter_bare_cases(args.case, args.m_list, args.custom_shapes))
    results: Dict[Tuple[str, str, int, int, int], float] = {}

    print(
        "config,shape,M,K,N,repeat,warmup,iters,mean_ms,median_ms,min_ms,max_ms,cv_pct,tflops"
    )
    old_config = os.environ.get("FP8_BLOCKWISE_SM120_FORCE_CONFIG")
    try:
        for config in configs:
            _set_force_config(config)
            for bench_case in bench_cases:
                run, flops = _make_bare_gemm_case(bench_case)
                means = [
                    _measure_ms(run, args.warmup, args.iters)[0]
                    for _ in range(args.repeat)
                ]
                mean_ms = statistics.mean(means)
                median_ms = statistics.median(means)
                min_ms = min(means)
                max_ms = max(means)
                stdev = statistics.stdev(means) if len(means) > 1 else 0.0
                cv_pct = stdev / mean_ms * 100.0 if mean_ms > 0 else 0.0
                tflops = flops / (mean_ms / 1000.0) / 1e12
                key = (config, bench_case.name, bench_case.m, bench_case.k, bench_case.n)
                results[key] = mean_ms
                print(
                    f"{config},{bench_case.name},{bench_case.m},{bench_case.k},{bench_case.n},"
                    f"{args.repeat},{args.warmup},{args.iters},"
                    f"{mean_ms:.6f},{median_ms:.6f},{min_ms:.6f},"
                    f"{max_ms:.6f},{cv_pct:.3f},{tflops:.2f}"
                )
    finally:
        _restore_force_config(old_config)

    if len(configs) < 2:
        return

    baseline = configs[0]
    print("\nspeedup_vs_" + baseline)
    print("shape,M,K,N," + ",".join(configs[1:]))
    for bench_case in bench_cases:
        baseline_ms = results.get(
            (baseline, bench_case.name, bench_case.m, bench_case.k, bench_case.n)
        )
        speedups = []
        for config in configs[1:]:
            current_ms = results.get(
                (config, bench_case.name, bench_case.m, bench_case.k, bench_case.n)
            )
            if baseline_ms is None or current_ms is None:
                speedups.append("nan")
            else:
                speedups.append(f"{baseline_ms / current_ms:.4f}")
        print(
            f"{bench_case.name},{bench_case.m},{bench_case.k},{bench_case.n},"
            + ",".join(speedups)
        )


def _make_rms_case(shape: ProjectionShape, m: int) -> Tuple[
    Callable[[], torch.Tensor], Callable[[], torch.Tensor]
]:
    linear = _make_blockwise_linear(shape.k, shape.n)
    rms_weight = torch.ones(shape.k, device="cuda", dtype=torch.bfloat16)
    rms = RMSNorm(rms_weight)
    hidden_states = torch.randn(m, shape.k, device="cuda", dtype=torch.bfloat16)

    def baseline() -> torch.Tensor:
        normed = rms(hidden_states)
        return linear(normed)

    def fused() -> torch.Tensor:
        quantized = linear.quantize_rmsnorm(rms, hidden_states)
        if quantized is None:
            raise RuntimeError("RMSNorm fused quantization is not available")
        return linear(quantized)

    return baseline, fused


def rms_stress(args: argparse.Namespace) -> None:
    shape = PROJECTION_SHAPES[args.case]
    _print_header("Operator stress: RMSNorm+FP8 quant + Linear GEMM")
    print(f"projection={shape.name} M={args.m} K={shape.k} N={shape.n}")
    baseline, fused = _make_rms_case(shape, args.m)

    with torch.inference_mode():
        diff = (baseline().float() - fused().float()).abs()
        print(
            f"correctness sanity: diff max={diff.max().item():.6g} "
            f"mean={diff.mean().item():.6g}"
        )
        _run_selected_modes(args.mode, baseline, fused, args.duration, args.report_interval)


def _make_silu_case(m: int) -> Tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    shape = PROJECTION_SHAPES["down"]
    linear = _make_blockwise_linear(shape.k, shape.n)
    gate_up = torch.randn(m, shape.k * 2, device="cuda", dtype=torch.bfloat16)

    def baseline() -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return linear(F.silu(gate) * up)

    def fused() -> torch.Tensor:
        return linear(linear.quantize_fused_silu_and_mul(gate_up))

    return baseline, fused


def silu_stress(args: argparse.Namespace) -> None:
    shape = PROJECTION_SHAPES["down"]
    _print_header("Operator stress: SiLU+mul+FP8 quant + down_proj GEMM")
    print(f"shape: M={args.m} K={shape.k} N={shape.n}")
    baseline, fused = _make_silu_case(args.m)

    with torch.inference_mode():
        diff = (baseline().float() - fused().float()).abs()
        print(
            f"correctness sanity: diff max={diff.max().item():.6g} "
            f"mean={diff.mean().item():.6g}"
        )
        _run_selected_modes(args.mode, baseline, fused, args.duration, args.report_interval)


def _make_gated_mlp_case(
    m: int, hidden_size: int, intermediate_size: int
) -> Tuple[
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
]:
    gate_up = _make_blockwise_linear(hidden_size, intermediate_size * 2)
    scale128_interleaved_gate_up = _make_scale128_interleaved_gate_up(gate_up)
    down = _make_blockwise_linear(intermediate_size, hidden_size)
    x = torch.randn(m, hidden_size, device="cuda", dtype=torch.bfloat16)

    def baseline() -> torch.Tensor:
        gate_up_out = gate_up(x)
        gate, up = gate_up_out.chunk(2, dim=-1)
        return down(F.silu(gate) * up)

    def silu_quant() -> torch.Tensor:
        gate_up_out = gate_up(x)
        return down(down.quantize_fused_silu_and_mul(gate_up_out))

    def activated_tail_ref() -> torch.Tensor:
        gate_up_out = gate_up(x)
        gate, up = gate_up_out.chunk(2, dim=-1)
        return down(down._quantize_input(F.silu(gate) * up))

    def scale128_interleaved_staged() -> torch.Tensor:
        gate_up_out = _restore_gate_up_from_scale128_interleaved(
            scale128_interleaved_gate_up(x)
        )
        gate, up = gate_up_out.chunk(2, dim=-1)
        return down(down._quantize_input(F.silu(gate) * up))

    def staged_producer() -> torch.Tensor:
        if cutlass_gated_mlp_staged_producer_sm120_fp8 is None:
            raise RuntimeError(
                "cutlass_gated_mlp_staged_producer_sm120_fp8 is not available"
            )
        data, scale = cutlass_gated_mlp_staged_producer_sm120_fp8(
            x, gate_up.weight, gate_up.weight_scales, None
        )
        return down(make_sm120_fp8_activation(data, scale, x.dtype, data.shape))

    def semi_tail_emulation() -> torch.Tensor:
        if cutlass_gated_mlp_semi_tail_emulation_sm120_fp8 is None:
            raise RuntimeError(
                "cutlass_gated_mlp_semi_tail_emulation_sm120_fp8 is not available"
            )
        data, scale = cutlass_gated_mlp_semi_tail_emulation_sm120_fp8(
            x, gate_up.weight, gate_up.weight_scales, None
        )
        return down(make_sm120_fp8_activation(data, scale, x.dtype, data.shape))

    def reference_producer() -> torch.Tensor:
        if cutlass_gated_mlp_producer_sm120_fp8 is None:
            raise RuntimeError("cutlass_gated_mlp_producer_sm120_fp8 is not available")
        data, scale = cutlass_gated_mlp_producer_sm120_fp8(
            x, gate_up.weight, gate_up.weight_scales, None
        )
        return down(make_sm120_fp8_activation(data, scale, x.dtype, data.shape))

    return (
        baseline,
        silu_quant,
        activated_tail_ref,
        scale128_interleaved_staged,
        staged_producer,
        semi_tail_emulation,
        reference_producer,
    )


def gated_mlp_stress(args: argparse.Namespace) -> None:
    _print_header("Operator stress: gated MLP FP8 blockwise paths")
    m_values = _parse_m_list(args.m_list)
    for m in m_values:
        gate_up_mib = m * args.intermediate_size * 2 * 2 / 1024 / 1024
        activated_mib = m * args.intermediate_size * 2 / 1024 / 1024
        print(
            f"\nshape: M={m} hidden={args.hidden_size} "
            f"intermediate={args.intermediate_size}"
        )
        print(
            f"bf16 intermediates: gate_up={gate_up_mib:.2f} MiB, "
            f"activated={activated_mib:.2f} MiB"
        )
        baseline, silu_quant, _, _, _, _, _ = _make_gated_mlp_case(
            m, args.hidden_size, args.intermediate_size
        )
        with torch.inference_mode():
            baseline_out = baseline().float()
            diff = (baseline_out - silu_quant().float()).abs()
            print(
                f"silu_quant correctness sanity: diff max={diff.max().item():.6g} "
                f"mean={diff.mean().item():.6g}"
            )
            _run_selected_modes(
                args.mode,
                baseline,
                silu_quant,
                args.duration,
                args.report_interval,
                optimized_label="partial_silu_quant",
            )


def gated_staged_producer_compare(args: argparse.Namespace) -> None:
    _print_header("Compare gated MLP producer boundaries")
    print(
        "# reference_producer directly writes FP8 [M,I] data plus scales but "
        "uses a tiled CUDA reference body; use --paths reference_producer on "
        "small M for dataflow/correctness, not speed."
    )
    print(
        "M,hidden,intermediate,path,mean_ms,p50_ms,min_ms,"
        "diff_vs_silu_max,diff_vs_silu_mean,diff_vs_tail_max,diff_vs_tail_mean"
    )
    for m in _parse_m_list(args.m_list):
        (
            baseline,
            silu_quant,
            activated_tail_ref,
            scale128_interleaved_staged,
            staged_producer,
            semi_tail_emulation,
            reference_producer,
        ) = _make_gated_mlp_case(m, args.hidden_size, args.intermediate_size)
        with torch.inference_mode():
            silu_ref = silu_quant().float()
            tail_ref = activated_tail_ref().float()
            path_fns = {
                "baseline": baseline,
                "silu_quant": silu_quant,
                "activated_tail_ref": activated_tail_ref,
                "scale128_interleaved_staged": scale128_interleaved_staged,
                "staged_producer": staged_producer,
                "semi_tail_emulation": semi_tail_emulation,
                "reference_producer": reference_producer,
            }
            for name in _parse_path_list(args.paths, GATED_PRODUCER_PATHS):
                fn = path_fns[name]
                out = fn().float()
                diff_silu = (silu_ref - out).abs()
                diff_tail = (tail_ref - out).abs()
                mean_ms, p50_ms, min_ms = _measure_ms(fn, args.warmup, args.iters)
                print(
                    f"{m},{args.hidden_size},{args.intermediate_size},{name},"
                    f"{mean_ms:.6f},{p50_ms:.6f},{min_ms:.6f},"
                    f"{diff_silu.max().item():.6g},{diff_silu.mean().item():.6g},"
                    f"{diff_tail.max().item():.6g},{diff_tail.mean().item():.6g}"
                )


def gated_mlp_breakdown(args: argparse.Namespace) -> None:
    _print_header("Operator breakdown: gated MLP FP8 blockwise staged path")
    print(
        "M,hidden,intermediate,stage,mean_ms,p50_ms,min_ms,"
        "relative_to_silu_quant_path_pct"
    )
    for m in _parse_m_list(args.m_list):
        gate_up = _make_blockwise_linear(args.hidden_size, args.intermediate_size * 2)
        down = _make_blockwise_linear(args.intermediate_size, args.hidden_size)
        x = torch.randn(m, args.hidden_size, device="cuda", dtype=torch.bfloat16)
        gate_up_out = gate_up(x)
        quantized = down.quantize_fused_silu_and_mul(gate_up_out)

        def run_gate_up() -> torch.Tensor:
            return gate_up(x)

        def run_silu_quant():
            return down.quantize_fused_silu_and_mul(gate_up_out)

        def run_down_quantized() -> torch.Tensor:
            return down(quantized)

        def run_silu_quant_path() -> torch.Tensor:
            return down(down.quantize_fused_silu_and_mul(gate_up(x)))

        stages = [
            ("gate_up_project", run_gate_up),
            ("silu_mul_quant", run_silu_quant),
            ("down_quantized_gemm", run_down_quantized),
            ("silu_quant_path", run_silu_quant_path),
        ]
        measured: Dict[str, Tuple[float, float, float]] = {}
        for name, fn in stages:
            measured[name] = _measure_ms(fn, args.warmup, args.iters)

        path_ms = measured["silu_quant_path"][0]
        for name, (mean_ms, p50_ms, min_ms) in measured.items():
            share = mean_ms / path_ms * 100.0 if path_ms > 0 else float("nan")
            print(
                f"{m},{args.hidden_size},{args.intermediate_size},{name},"
                f"{mean_ms:.6f},{p50_ms:.6f},{min_ms:.6f},{share:.2f}"
            )


def gated_fusion_opportunity(args: argparse.Namespace) -> None:
    _print_header("Estimate article-style gated MLP fusion opportunity")
    print(
        "M,hidden,intermediate,silu_quant_path_ms,gate_up_ms,"
        "silu_mul_quant_ms,activated_quant_ms,down_quantized_ms,"
        "gate_up_bf16_mib,activated_bf16_mib,activated_fp8_mib,"
        "semi_true_removable_stage_ms,full_fp8_tail_extra_ms,"
        "article_full_removable_stage_ms,article_upper_bound_path_ms,"
        "article_upper_bound_speedup"
    )
    for m in _parse_m_list(args.m_list):
        gate_up = _make_blockwise_linear(args.hidden_size, args.intermediate_size * 2)
        down = _make_blockwise_linear(args.intermediate_size, args.hidden_size)
        x = torch.randn(m, args.hidden_size, device="cuda", dtype=torch.bfloat16)
        gate_up_out = gate_up(x)
        gate, up = gate_up_out.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        quantized = down.quantize_fused_silu_and_mul(gate_up_out)

        def run_gate_up() -> torch.Tensor:
            return gate_up(x)

        def run_silu_quant():
            return down.quantize_fused_silu_and_mul(gate_up_out)

        def run_activated_quant():
            return down._quantize_input(activated)

        def run_down_quantized() -> torch.Tensor:
            return down(quantized)

        def run_silu_quant_path() -> torch.Tensor:
            return down(down.quantize_fused_silu_and_mul(gate_up(x)))

        gate_up_ms = _measure_ms(run_gate_up, args.warmup, args.iters)[0]
        silu_quant_ms = _measure_ms(run_silu_quant, args.warmup, args.iters)[0]
        activated_quant_ms = _measure_ms(run_activated_quant, args.warmup, args.iters)[0]
        down_ms = _measure_ms(run_down_quantized, args.warmup, args.iters)[0]
        path_ms = _measure_ms(run_silu_quant_path, args.warmup, args.iters)[0]

        gate_up_bf16_mib = m * args.intermediate_size * 2 * 2 / 1024 / 1024
        activated_bf16_mib = m * args.intermediate_size * 2 / 1024 / 1024
        activated_fp8_mib = m * args.intermediate_size / 1024 / 1024
        semi_true_removable_stage_ms = silu_quant_ms
        full_fp8_tail_extra_ms = max(0.0, activated_quant_ms - silu_quant_ms)
        article_full_removable_stage_ms = max(silu_quant_ms, activated_quant_ms)
        article_upper_bound_path_ms = max(
            0.0,
            path_ms - min(path_ms, article_full_removable_stage_ms),
        )
        speedup = path_ms / max(article_upper_bound_path_ms, 1e-9)
        print(
            f"{m},{args.hidden_size},{args.intermediate_size},"
            f"{path_ms:.6f},{gate_up_ms:.6f},{silu_quant_ms:.6f},"
            f"{activated_quant_ms:.6f},{down_ms:.6f},"
            f"{gate_up_bf16_mib:.2f},{activated_bf16_mib:.2f},"
            f"{activated_fp8_mib:.2f},{semi_true_removable_stage_ms:.6f},"
            f"{full_fp8_tail_extra_ms:.6f},{article_full_removable_stage_ms:.6f},"
            f"{article_upper_bound_path_ms:.6f},{speedup:.4f}"
        )


def _run_selected_modes(
    mode: str,
    baseline: Callable[[], torch.Tensor],
    fused: Callable[[], torch.Tensor],
    duration: float,
    report_interval: float,
    optimized_label: str = "fused",
) -> None:
    summary: Dict[str, Tuple[int, float, float]] = {}
    if mode in ("baseline", "both"):
        summary["baseline"] = _run_stress(
            "baseline", baseline, duration, report_interval
        )
    if mode in ("fused", "both"):
        summary["fused"] = _run_stress(optimized_label, fused, duration, report_interval)
    if "baseline" in summary and "fused" in summary:
        _, baseline_ops, baseline_ms = summary["baseline"]
        _, fused_ops, fused_ms = summary["fused"]
        print("\nsummary:")
        print(f"  baseline ops/s={baseline_ops:.2f} avg_ms/op={baseline_ms:.4f}")
        print(f"  {optimized_label:<8} ops/s={fused_ops:.2f} avg_ms/op={fused_ms:.4f}")
        print(f"  throughput speedup={fused_ops / baseline_ops:.3f}x")
        print(f"  latency speedup={baseline_ms / fused_ms:.3f}x")
        print(f"  saved_ms/op={baseline_ms - fused_ms:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bare = subparsers.add_parser("bare-gemm")
    bare.add_argument("--case", default="all", choices=["all", *PROJECTION_SHAPES])
    bare.add_argument("--m-list", default="768,1024,1536,2048")
    bare.add_argument(
        "--custom-shapes",
        help="Comma-separated MxKxN shapes. If set, --case/--m-list are ignored.",
    )
    bare.add_argument("--warmup", type=int, default=50)
    bare.add_argument("--iters", type=int, default=200)
    bare.set_defaults(func=bare_gemm)

    compare = subparsers.add_parser("compare-bare")
    compare.add_argument("--case", default="all", choices=["all", *PROJECTION_SHAPES])
    compare.add_argument("--m-list", default="768,1024,1536,2048")
    compare.add_argument(
        "--custom-shapes",
        help="Comma-separated MxKxN shapes. If set, --case/--m-list are ignored.",
    )
    compare.add_argument("--configs", default="legacy,auto")
    compare.add_argument("--repeat", type=int, default=10)
    compare.add_argument("--warmup", type=int, default=300)
    compare.add_argument("--iters", type=int, default=2000)
    compare.set_defaults(func=compare_bare)

    rms = subparsers.add_parser("rms-stress")
    rms.add_argument("--case", default="qkv", choices=["qkv", "o_proj", "gate_up"])
    rms.add_argument("--m", type=int, default=1024)
    rms.add_argument("--duration", type=float, default=30.0)
    rms.add_argument("--report-interval", type=float, default=5.0)
    rms.add_argument("--mode", default="both", choices=["baseline", "fused", "both"])
    rms.set_defaults(func=rms_stress)

    silu = subparsers.add_parser("silu-stress")
    silu.add_argument("--m", type=int, default=1536)
    silu.add_argument("--duration", type=float, default=30.0)
    silu.add_argument("--report-interval", type=float, default=5.0)
    silu.add_argument("--mode", default="both", choices=["baseline", "fused", "both"])
    silu.set_defaults(func=silu_stress)

    gated_mlp = subparsers.add_parser("gated-mlp-stress")
    gated_mlp.add_argument("--m-list", default="768,1024,1536,2048")
    gated_mlp.add_argument("--hidden-size", type=int, default=QWEN2_05B_HIDDEN)
    gated_mlp.add_argument(
        "--intermediate-size", type=int, default=QWEN2_05B_INTERMEDIATE
    )
    gated_mlp.add_argument("--duration", type=float, default=30.0)
    gated_mlp.add_argument("--report-interval", type=float, default=5.0)
    gated_mlp.add_argument(
        "--mode", default="both", choices=["baseline", "fused", "both"]
    )
    gated_mlp.set_defaults(func=gated_mlp_stress)

    gated_producer = subparsers.add_parser(
        "gated-staged-producer-compare",
        aliases=["gated-producer-compare"],
    )
    gated_producer.add_argument("--m-list", default="768,1024,1536,2048")
    gated_producer.add_argument("--hidden-size", type=int, default=QWEN2_05B_HIDDEN)
    gated_producer.add_argument(
        "--intermediate-size", type=int, default=QWEN2_05B_INTERMEDIATE
    )
    gated_producer.add_argument("--warmup", type=int, default=100)
    gated_producer.add_argument("--iters", type=int, default=1000)
    gated_producer.add_argument(
        "--paths",
        default=",".join(DEFAULT_GATED_PRODUCER_PATHS),
        help=(
            "Comma-separated producer paths, or all. reference_producer is a "
            "tiled CUDA full-output reference and is excluded by default."
        ),
    )
    gated_producer.set_defaults(func=gated_staged_producer_compare)

    gated_breakdown = subparsers.add_parser("gated-mlp-breakdown")
    gated_breakdown.add_argument("--m-list", default="768,1024,1536,2048")
    gated_breakdown.add_argument("--hidden-size", type=int, default=QWEN2_05B_HIDDEN)
    gated_breakdown.add_argument(
        "--intermediate-size", type=int, default=QWEN2_05B_INTERMEDIATE
    )
    gated_breakdown.add_argument("--warmup", type=int, default=100)
    gated_breakdown.add_argument("--iters", type=int, default=1000)
    gated_breakdown.set_defaults(func=gated_mlp_breakdown)

    gated_opportunity = subparsers.add_parser("gated-fusion-opportunity")
    gated_opportunity.add_argument("--m-list", default="768,1024,1536,2048")
    gated_opportunity.add_argument(
        "--hidden-size", type=int, default=QWEN2_05B_HIDDEN
    )
    gated_opportunity.add_argument(
        "--intermediate-size", type=int, default=QWEN2_05B_INTERMEDIATE
    )
    gated_opportunity.add_argument("--warmup", type=int, default=100)
    gated_opportunity.add_argument("--iters", type=int, default=1000)
    gated_opportunity.set_defaults(func=gated_fusion_opportunity)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
