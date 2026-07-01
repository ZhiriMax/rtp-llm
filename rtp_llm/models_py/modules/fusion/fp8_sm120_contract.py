"""Shared SM120 FP8 activation contract."""

import torch

from rtp_llm.models_py.modules.fusion.quant_activation import (
    GroupShape,
    QuantKey,
    QuantizedActivation,
    ScaleDesc,
    ScaleLayout,
)


SM120_FP8_INPUT_GROUP_SIZE = 128
SM120_FP8_INPUT_EPS = 1e-4
SM120_FP8_INPUT_QUANT_KEY = QuantKey(
    dtype=torch.float8_e4m3fn,
    scale=ScaleDesc(
        dtype=torch.float32,
        static=False,
        group_shape=GroupShape(1, SM120_FP8_INPUT_GROUP_SIZE),
    ),
    symmetric=True,
)
SM120_FP8_INPUT_SCALE_LAYOUT = ScaleLayout(
    column_major_scales=True,
    scale_tma_aligned=False,
    scale_ue8m0=False,
)


def make_sm120_fp8_activation(
    data: torch.Tensor,
    scale: torch.Tensor,
    orig_dtype: torch.dtype,
    orig_shape: torch.Size,
) -> QuantizedActivation:
    return QuantizedActivation(
        data,
        scale,
        orig_dtype,
        orig_shape,
        SM120_FP8_INPUT_QUANT_KEY,
        SM120_FP8_INPUT_SCALE_LAYOUT,
    )
