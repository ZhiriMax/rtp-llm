# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's model_executor/layers/fusion/quant_activation.py.

"""Typed contract for already-quantized activation tensors."""

from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Optional, Union

import torch


class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]
    PER_CHANNEL: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_channel(self) -> bool:
        return self.row == -1 and self.col == 1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)
GroupShape.PER_CHANNEL = GroupShape(-1, 1)


@dataclass(frozen=True)
class ScaleDesc:
    dtype: torch.dtype
    static: bool
    group_shape: GroupShape


@dataclass(frozen=True)
class QuantKey:
    dtype: torch.dtype
    scale: ScaleDesc
    scale2: Optional[ScaleDesc] = None
    symmetric: bool = True


@dataclass(frozen=True)
class ScaleLayout:
    # RTP-LLM keeps scale tensor layout separate from vLLM-style QuantKey
    # because some CUTLASS consumers require an exact memory layout.
    column_major_scales: bool = False
    scale_tma_aligned: bool = False
    scale_ue8m0: bool = False


@dataclass(frozen=True)
class QuantizedActivation:
    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey
    scale_layout: ScaleLayout


def as_quantized_activation(
    x: Union[torch.Tensor, QuantizedActivation],
    expected_key: QuantKey,
    expected_layout: ScaleLayout,
) -> Optional[QuantizedActivation]:
    if not isinstance(x, QuantizedActivation):
        return None
    if x.quant_key != expected_key:
        raise ValueError(
            f"QuantizedActivation key {x.quant_key} != consumer key {expected_key}"
        )
    if x.scale_layout != expected_layout:
        raise ValueError(
            f"QuantizedActivation scale layout {x.scale_layout} != consumer "
            f"layout {expected_layout}"
        )
    return x
