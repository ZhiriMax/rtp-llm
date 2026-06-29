"""SM120 FP8 blockwise GEMM operator microbenchmarks.

Examples:
  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 bare-gemm \
      --case all --m-list 768,1024,1536,2048

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 rms-stress \
      --case qkv --m 1024 --duration 30 --mode both

  python3 -m rtp_llm.tools.bench.bench_fp8_blockwise_sm120 silu-stress \
      --m 1536 --duration 30 --mode both
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.ops.compute_ops import cutlass_scaled_mm_blockwise_sm120_fp8


QWEN2_05B_HIDDEN = 896
QWEN2_05B_INTERMEDIATE = 4864
QWEN2_05B_QKV = 1152


@dataclass(frozen=True)
class ProjectionShape:
    name: str
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
    env_keys = [
        "FP8_BLOCKWISE_SM120_FORCE_CONFIG",
        "FP8_BLOCKWISE_SM120_DISPATCH_POLICY",
        "FP8_BLOCKWISE_SM120_PINGPONG_MAX_M",
        "FP8_BLOCKWISE_SM120_PINGPONG_MAX_N",
        "FP8_BLOCKWISE_SM120_PINGPONG_MIN_K_TILES",
        "FP8_BLOCKWISE_SM120_PINGPONG_MAX_K_TILES",
    ]
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


def _select_cases(case: str) -> Iterable[ProjectionShape]:
    if case == "all":
        return PROJECTION_SHAPES.values()
    if case not in PROJECTION_SHAPES:
        raise ValueError(f"unknown case={case}, expected one of all/{list(PROJECTION_SHAPES)}")
    return [PROJECTION_SHAPES[case]]


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
    m_values = _parse_m_list(args.m_list)
    for shape in _select_cases(args.case):
        linear = _make_blockwise_linear(shape.k, shape.n)
        for m in m_values:
            x = torch.randn(m, shape.k, device="cuda", dtype=torch.bfloat16)
            input_fp8, input_scales = linear._quantize_input(x)
            out = torch.empty(m, shape.n, device="cuda", dtype=torch.bfloat16)

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

            mean, p50, min_ms = _measure_ms(run, args.warmup, args.iters)
            tflops = (2.0 * m * shape.k * shape.n) / (mean / 1000.0) / 1e12
            print(
                f"{shape.name:<8} M={m:5d} K={shape.k:5d} N={shape.n:5d} "
                f"mean={mean:8.4f} ms p50={p50:8.4f} ms min={min_ms:8.4f} ms "
                f"{tflops:8.2f} TFLOPS"
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


def _run_selected_modes(
    mode: str,
    baseline: Callable[[], torch.Tensor],
    fused: Callable[[], torch.Tensor],
    duration: float,
    report_interval: float,
) -> None:
    summary: Dict[str, Tuple[int, float, float]] = {}
    if mode in ("baseline", "both"):
        summary["baseline"] = _run_stress(
            "baseline", baseline, duration, report_interval
        )
    if mode in ("fused", "both"):
        summary["fused"] = _run_stress("fused", fused, duration, report_interval)
    if "baseline" in summary and "fused" in summary:
        _, baseline_ops, baseline_ms = summary["baseline"]
        _, fused_ops, fused_ms = summary["fused"]
        print("\nsummary:")
        print(f"  baseline ops/s={baseline_ops:.2f} avg_ms/op={baseline_ms:.4f}")
        print(f"  fused    ops/s={fused_ops:.2f} avg_ms/op={fused_ms:.4f}")
        print(f"  throughput speedup={fused_ops / baseline_ops:.3f}x")
        print(f"  latency speedup={baseline_ms / fused_ms:.3f}x")
        print(f"  saved_ms/op={baseline_ms - fused_ms:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bare = subparsers.add_parser("bare-gemm")
    bare.add_argument("--case", default="all", choices=["all", *PROJECTION_SHAPES])
    bare.add_argument("--m-list", default="768,1024,1536,2048")
    bare.add_argument("--warmup", type=int, default=50)
    bare.add_argument("--iters", type=int, default=200)
    bare.set_defaults(func=bare_gemm)

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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
