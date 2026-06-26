"""CUDA FP8 PER_BLOCK GEMM for sm_120 family (consumer Blackwell).

Backend wraps the vLLM-ported `cutlass_scaled_mm_blockwise_sm120_fp8`
kernel (see `models_py/bindings/cuda/cutlass/cutlass_kernels/fp8_blockwise_sm120/`).
Selected by LinearFactory only when `is_sm12x()` is true; sm_9x / sm_10x
keep using DeepGEMM via `CudaFp8GEMMLinear`.
"""

import logging
import os
from collections import Counter
from typing import Optional, Protocol, Tuple, Union

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    rms_norm_per_block_quant_fp8,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.models_py.modules.fusion.quant_activation import (
    GroupShape,
    QuantKey,
    QuantizedActivation,
    ScaleLayout,
    ScaleDesc,
    as_quantized_activation,
)
from rtp_llm.models_py.utils.arch import is_cuda, is_sm12x
from rtp_llm.ops import HWKernelConfig

if is_cuda() and is_sm12x():
    from rtp_llm.ops.compute_ops import cutlass_scaled_mm_blockwise_sm120_fp8
else:
    cutlass_scaled_mm_blockwise_sm120_fp8 = None


_FP8_GEMM_SHAPE_TELEMETRY_ENABLED = (
    os.environ.get("ENABLE_FP8_GEMM_SHAPE_TELEMETRY", "0") == "1"
)
_FP8_GEMM_SHAPE_COUNTER = Counter()
_FP8_GEMM_SHAPE_TOTAL = 0


def _get_positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        logging.warning("Invalid %s=%r, fallback to %d", name, value, default)
        return default


_FP8_GEMM_SHAPE_TELEMETRY_INTERVAL = _get_positive_int_env(
    "FP8_GEMM_SHAPE_TELEMETRY_INTERVAL", 1000
)
_SM120_FP8_INPUT_QUANT_KEY = QuantKey(
    dtype=torch.float8_e4m3fn,
    scale=ScaleDesc(
        dtype=torch.float32,
        static=False,
        group_shape=GroupShape(1, 128),
    ),
    symmetric=True,
)
_SM120_FP8_INPUT_SCALE_LAYOUT = ScaleLayout(
    column_major_scales=True,
    scale_tma_aligned=False,
    scale_ue8m0=False,
)
_SM120_FP8_INPUT_GROUP_SIZE = _SM120_FP8_INPUT_QUANT_KEY.scale.group_shape.col
_SM120_FP8_INPUT_EPS = 1e-4


def _m_bucket(m: int) -> str:
    if m <= 8:
        return "1-8"
    if m <= 16:
        return "9-16"
    if m <= 32:
        return "17-32"
    if m <= 64:
        return "33-64"
    if m <= 128:
        return "65-128"
    if m <= 256:
        return "129-256"
    if m <= 512:
        return "257-512"
    if m <= 768:
        return "513-768"
    if m <= 1024:
        return "769-1024"
    if m <= 1536:
        return "1025-1536"
    if m <= 2048:
        return "1537-2048"
    if m <= 3072:
        return "2049-3072"
    if m <= 4096:
        return "3073-4096"
    return "4097+"


def _selected_sm120_config(m: int) -> str:
    if m <= 64:
        return "swap_ab"
    if m <= 256:
        return "pingpong"
    return "default"


def _record_fp8_gemm_shape(m: int, k: int, n: int) -> None:
    if not _FP8_GEMM_SHAPE_TELEMETRY_ENABLED:
        return

    global _FP8_GEMM_SHAPE_TOTAL
    selected_config = _selected_sm120_config(m)
    key = (_m_bucket(m), k, n, selected_config)
    _FP8_GEMM_SHAPE_COUNTER[key] += 1
    _FP8_GEMM_SHAPE_TOTAL += 1

    if _FP8_GEMM_SHAPE_TOTAL % _FP8_GEMM_SHAPE_TELEMETRY_INTERVAL != 0:
        return

    top_buckets = _FP8_GEMM_SHAPE_COUNTER.most_common(20)
    summary = "; ".join(
        f"m_bucket={m_bucket},k={k},n={n},config={config},count={count}"
        for (m_bucket, k, n, config), count in top_buckets
    )
    logging.info(
        "fp8_gemm_shape_telemetry total=%d top=%s",
        _FP8_GEMM_SHAPE_TOTAL,
        summary,
    )


class _RMSNormLike(Protocol):
    weight: torch.Tensor
    variance_epsilon: float


class CudaFp8VllmBlockwiseLinear(LinearBase):
    """CUDA FP8 PER_BLOCK Linear for sm_120 (RTX PRO 5000 / 5090).

    Scale layout (matches CUTLASS Sm120BlockwiseScaleConfig<1, 128, 128, MN, K>):
      - input_scales : (M, K//128), MN-major (M-stride=1, K-group-stride=M)
      - weight_scales: (N//128, K//128), K-major  (K-stride = 1)
    Input scales use column_major_scales=True, scale_tma_aligned=False
    because CUTLASS tile_atom_to_shape_SFA computes K-group stride as exactly
    M (no alignment padding).  scale_tma_aligned=True would pad to ceil4(M),
    causing a stride mismatch for non-multiple-of-4 M values.
    """

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if weight_scales is None or quant_config is None:
            return False
        if not is_sm12x():
            return False
        if weight.dtype != torch.float8_e4m3fn:
            return False
        # vLLM kernel wants float32 PER_BLOCK scales — UE8M0 (int32) is a
        # DeepGEMM-only encoding and is not supported here.
        if weight_scales.dtype != torch.float32:
            return False
        return quant_config.get_method() == "FP8_PER_BLOCK"

    @torch.inference_mode()
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        if cutlass_scaled_mm_blockwise_sm120_fp8 is None:
            raise RuntimeError(
                "cutlass_scaled_mm_blockwise_sm120_fp8 op is not available; "
                "this backend requires a cuda12_9_x86 build with -DENABLE_FP8_SM120."
            )

        self.weight = weight
        self.weight_scales = weight_scales
        self.input_scales = input_scales
        self.bias = bias

        if self.weight.dim() != 2 or self.weight_scales.dim() != 2:
            raise ValueError(
                f"Weight and weight scale must be 2D tensors, got weight dim "
                f"{self.weight.dim()} and weight scale dim {self.weight_scales.dim()}"
            )

        self.K, self.N = self.weight.shape
        self.scale_K, self.scale_N = self.weight_scales.shape
        self.weight = self.weight.reshape(self.N, self.K).contiguous()
        self.weight_scales = self.weight_scales.reshape(
            self.scale_N, self.scale_K
        ).contiguous()

        if (self.N + 127) // 128 != self.scale_N or (
            self.K + 127
        ) // 128 != self.scale_K:
            raise ValueError(
                f"Weight scale dim mismatch: N={self.N} scale_N={self.scale_N}, "
                f"K={self.K} scale_K={self.scale_K} (expected ceil_div by 128)"
            )

        if self.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Weight dtype must be float8_e4m3fn, got {self.weight.dtype}"
            )

        if self.bias is not None:
            if self.bias.dim() not in (1, 2):
                raise ValueError(
                    f"Bias dimension must be 1 or 2, got {self.bias.dim()}"
                )
            if self.bias.shape[-1] != self.N:
                raise ValueError(
                    f"Bias last dimension must be {self.N}, got {self.bias.shape[-1]}"
                )
            if self.bias.dim() == 2 and self.bias.shape[0] != 1:
                raise ValueError(
                    f"Bias first dimension must be 1, got {self.bias.shape[0]}"
                )
            if self.bias.dtype != torch.bfloat16:
                raise ValueError(f"Bias dtype must be bfloat16, got {self.bias.dtype}")

    def _quantize_input(
        self, input: torch.Tensor, fuse_silu_and_mul: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return sgl_per_token_group_quant_fp8(
            input,
            group_size=_SM120_FP8_INPUT_GROUP_SIZE,
            eps=_SM120_FP8_INPUT_EPS,
            column_major_scales=_SM120_FP8_INPUT_SCALE_LAYOUT.column_major_scales,
            scale_tma_aligned=_SM120_FP8_INPUT_SCALE_LAYOUT.scale_tma_aligned,
            scale_ue8m0=_SM120_FP8_INPUT_SCALE_LAYOUT.scale_ue8m0,
            fuse_silu_and_mul=fuse_silu_and_mul,
        )

    def quantize_fused_silu_and_mul(
        self, gate_up: torch.Tensor
    ) -> QuantizedActivation:
        if gate_up.dtype != torch.bfloat16:
            raise ValueError(
                f"Input tensor dtype must be bfloat16, got {gate_up.dtype}"
            )
        if gate_up.dim() != 2:
            raise ValueError(
                f"Input tensor dimension must be 2, got {gate_up.dim()}D tensor"
            )
        if gate_up.shape[-1] != self.K * 2:
            raise ValueError(
                f"Fused SiLU input inner dimension expected to be {self.K * 2}, "
                f"got {gate_up.shape[-1]}"
            )
        if self.K % _SM120_FP8_INPUT_GROUP_SIZE != 0:
            raise ValueError(
                f"Fused SiLU output dimension {self.K} must be divisible by "
                f"group size {_SM120_FP8_INPUT_GROUP_SIZE}"
            )
        input_fp8, input_scales = self._quantize_input(
            gate_up, fuse_silu_and_mul=True
        )
        return QuantizedActivation(
            input_fp8,
            input_scales,
            gate_up.dtype,
            torch.Size((gate_up.shape[0], self.K)),
            _SM120_FP8_INPUT_QUANT_KEY,
            _SM120_FP8_INPUT_SCALE_LAYOUT,
        )

    def quantize_rmsnorm(
        self, rmsnorm: _RMSNormLike, hidden_states: torch.Tensor
    ) -> QuantizedActivation:
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                f"Input tensor dtype must be bfloat16, got {hidden_states.dtype}"
            )
        if hidden_states.dim() != 2:
            raise ValueError(
                f"Input tensor dimension must be 2, got {hidden_states.dim()}D tensor"
            )
        if hidden_states.shape[-1] != self.K:
            raise ValueError(
                f"Input tensor inner dimension expected to be {self.K}, got "
                f"{hidden_states.shape[-1]}"
            )
        input_fp8, input_scales = rms_norm_per_block_quant_fp8(
            hidden_states,
            rmsnorm.weight.data,
            rmsnorm.variance_epsilon,
            group_size=_SM120_FP8_INPUT_GROUP_SIZE,
            quant_eps=_SM120_FP8_INPUT_EPS,
            column_major_scales=_SM120_FP8_INPUT_SCALE_LAYOUT.column_major_scales,
            scale_tma_aligned=_SM120_FP8_INPUT_SCALE_LAYOUT.scale_tma_aligned,
            scale_ue8m0=_SM120_FP8_INPUT_SCALE_LAYOUT.scale_ue8m0,
        )
        return QuantizedActivation(
            input_fp8,
            input_scales,
            hidden_states.dtype,
            hidden_states.shape,
            _SM120_FP8_INPUT_QUANT_KEY,
            _SM120_FP8_INPUT_SCALE_LAYOUT,
        )

    def _forward_quantized(
        self, input_fp8: torch.Tensor, input_scales: torch.Tensor
    ) -> torch.Tensor:
        if input_fp8.dim() != 2:
            raise ValueError(
                f"Quantized input dimension must be 2, got {input_fp8.dim()}D tensor"
            )
        M, K = input_fp8.shape
        if K != self.K:
            raise ValueError(
                f"Quantized input inner dimension expected to be {self.K}, got {K}"
            )
        if input_fp8.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Quantized input dtype must be float8_e4m3fn, got {input_fp8.dtype}"
            )
        if input_scales.device != input_fp8.device:
            raise ValueError(
                f"Input scale device {input_scales.device} must match input device "
                f"{input_fp8.device}"
            )
        if input_scales.dtype != torch.float32:
            raise ValueError(
                f"Input scale dtype must be float32, got {input_scales.dtype}"
            )
        expected_scale_shape = (M, K // _SM120_FP8_INPUT_GROUP_SIZE)
        if input_scales.shape != expected_scale_shape:
            raise ValueError(
                f"Input scale shape expected to be {expected_scale_shape}, got "
                f"{tuple(input_scales.shape)}"
            )
        if _SM120_FP8_INPUT_SCALE_LAYOUT.column_major_scales:
            if input_scales.stride(-2) != 1 or input_scales.stride(-1) < M:
                raise ValueError(
                    "Column-major input scales must have token stride 1 and "
                    "K-group stride at least M"
                )
        elif (
            input_scales.stride(-1) != 1
            or input_scales.stride(-2) < expected_scale_shape[-1]
        ):
            raise ValueError(
                "Row-major input scales must have K-group stride 1 and token "
                "stride at least K-group count"
            )
        _record_fp8_gemm_shape(M, K, self.N)

        output = torch.empty(M, self.N, dtype=torch.bfloat16, device=input_fp8.device)
        # Bias is fused into the GEMM epilogue (per-output-channel add) instead
        # of a separate elementwise add kernel.
        bias = None
        if self.bias is not None:
            bias = self.bias.reshape(-1).to(dtype=output.dtype, device=output.device)
        cutlass_scaled_mm_blockwise_sm120_fp8(
            output,
            input_fp8,
            self.weight,
            input_scales,
            self.weight_scales,
            bias,
        )
        return output

    def forward(self, input: Union[torch.Tensor, QuantizedActivation]) -> torch.Tensor:
        quantized = as_quantized_activation(
            input, _SM120_FP8_INPUT_QUANT_KEY, _SM120_FP8_INPUT_SCALE_LAYOUT
        )
        if quantized is not None:
            return self._forward_quantized(quantized.data, quantized.scale)

        if input.dtype != torch.bfloat16:
            raise ValueError(f"Input tensor dtype must be bfloat16, got {input.dtype}")
        if input.dim() != 2:
            raise ValueError(
                f"Input tensor dimension must be 2, got {input.dim()}D tensor"
            )
        _, K = input.shape
        if K != self.K:
            raise ValueError(
                f"Input tensor inner dimension expected to be {self.K}, got {K}"
            )
        input_fp8, input_scales = self._quantize_input(input)
        return self._forward_quantized(input_fp8, input_scales)
