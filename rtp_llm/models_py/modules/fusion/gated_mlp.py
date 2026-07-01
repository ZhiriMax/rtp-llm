"""Gated MLP fusion helpers.

This module owns MLP-level fusion policy. Generic Linear implementations should
not need to know Qwen-style gate/up semantics.
"""

import os
from typing import Callable, Optional

import torch
from torch import nn

from rtp_llm.models_py.modules.fusion.fp8_sm120_contract import (
    SM120_FP8_INPUT_GROUP_SIZE,
    make_sm120_fp8_activation,
)


FUSION_MODE_ENV = "RTP_LLM_DENSE_MLP_FUSION"
FUSION_OFF = "off"
FUSION_SILU_QUANT = "silu_quant"
# Developer-only direct-call modes. Keep them out of FUSION_MODES until the
# producer path is validated for serving.
FUSION_GATE_UP_STAGED_PRODUCER = "gate_up_staged_producer"
FUSION_GATE_UP_PRODUCER = "gate_up_producer"
FUSION_MODES = {
    FUSION_OFF,
    FUSION_SILU_QUANT,
}
GATE_UP_STAGED_PRODUCER_OP_NAME = "cutlass_gated_mlp_staged_producer_sm120_fp8"
GATE_UP_PRODUCER_OP_NAME = "cutlass_gated_mlp_producer_sm120_fp8"


def fusion_mode() -> str:
    mode = os.environ.get(FUSION_MODE_ENV)
    if mode is None:
        return (
            FUSION_SILU_QUANT
            if os.environ.get("ENABLE_DENSE_SILU_MUL_QUANT_FUSION", "0") == "1"
            else FUSION_OFF
        )
    mode = mode.strip().lower()
    if mode not in FUSION_MODES:
        raise ValueError(
            f"{FUSION_MODE_ENV} must be one of {sorted(FUSION_MODES)}, got {mode!r}"
        )
    return mode


def try_staged_silu_quant(
    up_proj: nn.Module,
    down_proj: nn.Module,
    x: torch.Tensor,
    is_gated: bool,
) -> Optional[torch.Tensor]:
    if not is_gated:
        return None
    quantize_fused = getattr(down_proj, "quantize_fused_silu_and_mul", None)
    if quantize_fused is None:
        return None
    return down_proj(quantize_fused(up_proj(x)))


def _compute_op(name: str) -> Optional[Callable]:
    try:
        import rtp_llm.ops.compute_ops as compute_ops
    except Exception:
        return None
    return getattr(compute_ops, name, None)


def _gate_up_staged_producer_op() -> Optional[Callable]:
    return _compute_op(GATE_UP_STAGED_PRODUCER_OP_NAME)


def _gate_up_producer_op(op_name: str) -> Optional[Callable]:
    if op_name == GATE_UP_STAGED_PRODUCER_OP_NAME:
        return _gate_up_staged_producer_op()
    return _compute_op(op_name)


def _is_gate_up_producer_compatible(
    up_proj: nn.Module,
    down_proj: nn.Module,
    x: torch.Tensor,
) -> bool:
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    if x.shape[0] == 0:
        return False
    if not all(
        hasattr(up_proj, name) for name in ("weight", "weight_scales", "K", "N")
    ):
        return False
    if not hasattr(down_proj, "K"):
        return False
    if not hasattr(down_proj, "quantize_fused_silu_and_mul"):
        return False
    if up_proj.K != x.shape[-1]:
        return False
    if up_proj.N != down_proj.K * 2:
        return False
    if up_proj.K % SM120_FP8_INPUT_GROUP_SIZE != 0:
        return False
    if up_proj.weight.shape != (up_proj.N, up_proj.K):
        return False
    if up_proj.weight.dtype != torch.float8_e4m3fn:
        return False
    if up_proj.weight.device != x.device or not up_proj.weight.is_contiguous():
        return False
    if up_proj.weight_scales.dtype != torch.float32:
        return False
    if up_proj.weight_scales.device != x.device:
        return False
    expected_weight_scale_shape = (
        (up_proj.N + SM120_FP8_INPUT_GROUP_SIZE - 1) // SM120_FP8_INPUT_GROUP_SIZE,
        up_proj.K // SM120_FP8_INPUT_GROUP_SIZE,
    )
    if tuple(up_proj.weight_scales.shape) != expected_weight_scale_shape:
        return False
    if up_proj.weight_scales.stride(-2) != expected_weight_scale_shape[-1]:
        return False
    if up_proj.weight_scales.stride(-1) != 1:
        return False
    return down_proj.K % SM120_FP8_INPUT_GROUP_SIZE == 0


def _check_gate_up_producer_result(
    op_name: str,
    data: torch.Tensor,
    scale: torch.Tensor,
    expected_shape: tuple,
    expected_scale_shape: tuple,
    scale_stride_m: int,
) -> None:
    if data.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{op_name} data dtype must be float8_e4m3fn")
    if not data.is_contiguous():
        raise ValueError(f"{op_name} data must be contiguous")
    if scale.dtype != torch.float32:
        raise ValueError(f"{op_name} scale dtype must be float32")
    if scale.device != data.device:
        raise ValueError(f"{op_name} scale must be on the data device")
    if tuple(data.shape) != expected_shape:
        raise ValueError(
            f"{op_name} data shape must be {expected_shape}, got "
            f"{tuple(data.shape)}"
        )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"{op_name} scale shape must be {expected_scale_shape}, got {tuple(scale.shape)}"
        )
    if scale.stride(-2) != 1 or scale.stride(-1) != scale_stride_m:
        raise ValueError(f"{op_name} scale must be column-major over tokens")


def try_gate_up_producer(
    op_name: str,
    up_proj: nn.Module,
    down_proj: nn.Module,
    x: torch.Tensor,
    is_gated: bool,
) -> Optional[torch.Tensor]:
    if not is_gated:
        return None
    op = _gate_up_producer_op(op_name)
    if op is None:
        return None
    if not _is_gate_up_producer_compatible(up_proj, down_proj, x):
        return None

    bias = getattr(up_proj, "bias", None)
    if bias is not None:
        if bias.dtype != x.dtype or bias.device != x.device or bias.numel() != up_proj.N:
            return None
        bias = bias.reshape(-1).contiguous()
    data, scale = op(x, up_proj.weight, up_proj.weight_scales, bias)
    expected_shape = (x.shape[0], down_proj.K)
    expected_scale_shape = (x.shape[0], down_proj.K // SM120_FP8_INPUT_GROUP_SIZE)
    _check_gate_up_producer_result(
        op_name,
        data,
        scale,
        expected_shape,
        expected_scale_shape,
        x.shape[0],
    )
    quantized = make_sm120_fp8_activation(
        data,
        scale,
        x.dtype,
        data.shape,
    )
    return down_proj(quantized)


def try_gate_up_staged_producer(
    up_proj: nn.Module,
    down_proj: nn.Module,
    x: torch.Tensor,
    is_gated: bool,
) -> Optional[torch.Tensor]:
    return try_gate_up_producer(
        GATE_UP_STAGED_PRODUCER_OP_NAME, up_proj, down_proj, x, is_gated
    )


def _require_gate_up_producer(
    mode: str,
    op_name: str,
    up_proj: nn.Module,
    down_proj: nn.Module,
    x: torch.Tensor,
    is_gated: bool,
) -> torch.Tensor:
    output = try_gate_up_producer(op_name, up_proj, down_proj, x, is_gated)
    if output is None:
        raise RuntimeError(
            f"{mode} requires {op_name} and SM120 FP8 gated MLP-compatible layers"
        )
    return output


def try_gated_mlp_fusion(
    mode: str,
    up_proj: nn.Module,
    down_proj: nn.Module,
    x: torch.Tensor,
    is_gated: bool,
) -> Optional[torch.Tensor]:
    if mode == FUSION_OFF:
        return None
    if mode == FUSION_SILU_QUANT:
        return try_staged_silu_quant(up_proj, down_proj, x, is_gated)
    if mode == FUSION_GATE_UP_STAGED_PRODUCER:
        return _require_gate_up_producer(
            mode,
            GATE_UP_STAGED_PRODUCER_OP_NAME,
            up_proj,
            down_proj,
            x,
            is_gated,
        )
    if mode == FUSION_GATE_UP_PRODUCER:
        return _require_gate_up_producer(
            mode,
            GATE_UP_PRODUCER_OP_NAME,
            up_proj,
            down_proj,
            x,
            is_gated,
        )
    raise ValueError(f"unsupported gated MLP fusion mode: {mode}")
