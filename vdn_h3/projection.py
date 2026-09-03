"""Precision-aware execution of the dominant VDN ``to_out_linear`` projection.

The H3 backbone commonly used in ComfyUI is already quantized. Leaving the extra VDN
projection in BF16 makes the hybrid model pay a large dense GEMM that the native model
has already eliminated. This module lets the VDN projection match the active inference
family while keeping the recurrent state, gates and statistics in their trained
BF16/FP32 islands.

Supported storage:

* ``bf16``: checkpoint weight as released;
* ``fp8``: OpenVDN-style E4M3 projection;
* ``int8``: stock ComfyUI TensorWiseINT8 + ConvRot, using the same comfy-kitchen kernel
  family as an INT8/ConvRot MiniMax-H3 base.

FP8 and INT8 deliberately share the existing quantized projection storage keys. The
storage layer already preserves the quantized tensor dtype and FP32 scale; dispatch is
based on the actual tensor dtype, so streaming/residency logic stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .fp8 import (
    BF16_WEIGHT_KEY,
    DEFAULT_SKIP_END_BLOCKS,
    FP8_SCALE_KEY,
    FP8_WEIGHT_KEY,
    FP8ProjectionInfo,
    fp8_supported,
    prepare_projection_maps as prepare_fp8_projection_maps,
    project as project_fp8,
)


# Aliases to the generic quantized-storage slot already understood by weights.py.
INT8_WEIGHT_KEY = FP8_WEIGHT_KEY
INT8_SCALE_KEY = FP8_SCALE_KEY
INT8_CONVROT_GROUP = 256
PROJECTION_PRECISIONS = ("bf16", "fp8", "int8")
QUANTIZED_PROJECTION_KEYS = frozenset({FP8_WEIGHT_KEY})
PROJECTION_SCALE_KEYS = frozenset({FP8_SCALE_KEY})


@dataclass(frozen=True)
class INT8ProjectionInfo:
    original_bytes: int
    quantized_bytes: int
    quantized_blocks: int
    bf16_blocks: int
    skip_end_blocks: int
    convrot: bool
    convrot_groupsize: int
    per_channel: bool = True


def _comfy_int8_components():
    try:
        import comfy.quant_ops as quant_ops
    except Exception as exc:  # pragma: no cover - depends on ComfyUI runtime
        raise RuntimeError("ComfyUI quant_ops/comfy-kitchen is required for INT8 VDN projection") from exc
    required = ("QuantizedTensor", "TensorWiseINT8Layout", "ck")
    missing = [name for name in required if not hasattr(quant_ops, name)]
    if missing:
        raise RuntimeError(f"ComfyUI INT8 runtime is missing {', '.join(missing)}")
    return quant_ops


def _quantize_int8_weight(
    weight: torch.Tensor,
    device: torch.device,
    *,
    convrot: bool,
    convrot_groupsize: int,
):
    quant_ops = _comfy_int8_components()
    use_convrot = bool(convrot and weight.shape[1] % int(convrot_groupsize) == 0)
    source = weight.detach().to(device=device, dtype=torch.bfloat16, non_blocking=False).contiguous()
    qweight = quant_ops.QuantizedTensor.from_float(
        source,
        "TensorWiseINT8Layout",
        scale="recalculate",
        stochastic_rounding=0,
        per_channel=True,
        convrot=use_convrot,
        convrot_groupsize=int(convrot_groupsize),
    )
    qdata, scale = quant_ops.TensorWiseINT8Layout.get_plain_tensors(qweight)
    return (
        qdata.detach().to(device="cpu", dtype=torch.int8).contiguous(),
        scale.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        use_convrot,
    )


def int8_supported(
    device: torch.device | str,
    *,
    convrot: bool = True,
    convrot_groupsize: int = INT8_CONVROT_GROUP,
    probe: bool = True,
) -> bool:
    try:
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return False
        quant_ops = _comfy_int8_components()
        if not probe:
            return True
        width = int(convrot_groupsize) if convrot else 256
        x = torch.randn(8, width, device=device, dtype=torch.bfloat16)
        w = torch.randn(32, width, device=device, dtype=torch.bfloat16)
        qweight = quant_ops.QuantizedTensor.from_float(
            w,
            "TensorWiseINT8Layout",
            scale="recalculate",
            stochastic_rounding=0,
            per_channel=True,
            convrot=bool(convrot),
            convrot_groupsize=int(convrot_groupsize),
        )
        qdata, scale = quant_ops.TensorWiseINT8Layout.get_plain_tensors(qweight)
        out = quant_ops.ck.int8_linear(
            x,
            qdata,
            scale,
            None,
            x.dtype,
            convrot=bool(convrot),
            convrot_groupsize=int(convrot_groupsize),
        )
        return tuple(out.shape) == (8, 32)
    except Exception:
        return False


def prepare_int8_projection_maps(
    branch_maps: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    *,
    skip_end_blocks: int = 0,
    convrot: bool = True,
    convrot_groupsize: int = INT8_CONVROT_GROUP,
):
    """Quantize VDN output projections to stock-Comfy INT8/ConvRot CPU masters."""
    device = torch.device(device)
    if not int8_supported(
        device,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        probe=True,
    ):
        raise RuntimeError(f"INT8/ConvRot VDN projection is unavailable on {device}")
    if isinstance(skip_end_blocks, bool) or not isinstance(skip_end_blocks, int) or skip_end_blocks < 0:
        raise ValueError("skip_end_blocks must be a non-negative integer")
    count = len(branch_maps)
    if count == 0:
        raise ValueError("branch_maps cannot be empty")
    if skip_end_blocks * 2 >= count and skip_end_blocks:
        raise ValueError(
            f"skip_end_blocks={skip_end_blocks} leaves no INT8 interior blocks for {count} branches"
        )

    transformed = []
    original_bytes = quantized_bytes = 0
    quantized_blocks = bf16_blocks = 0
    used_convrot = True
    for index, block in enumerate(branch_maps):
        if BF16_WEIGHT_KEY not in block:
            raise KeyError(f"VDN block {index} has no {BF16_WEIGHT_KEY!r}")
        weight = block[BF16_WEIGHT_KEY]
        original_bytes += int(weight.numel() * weight.element_size())
        keep_bf16 = index < skip_end_blocks or index >= count - skip_end_blocks
        if keep_bf16:
            transformed.append(dict(block))
            quantized_bytes += int(weight.numel() * weight.element_size())
            bf16_blocks += 1
            continue
        qdata, scale, block_convrot = _quantize_int8_weight(
            weight,
            device,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )
        copied = dict(block)
        copied.pop(BF16_WEIGHT_KEY)
        copied[FP8_WEIGHT_KEY] = qdata
        copied[FP8_SCALE_KEY] = scale
        transformed.append(copied)
        quantized_bytes += int(
            qdata.numel() * qdata.element_size() + scale.numel() * scale.element_size()
        )
        quantized_blocks += 1
        used_convrot = used_convrot and block_convrot

    return tuple(transformed), INT8ProjectionInfo(
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        quantized_blocks=quantized_blocks,
        bf16_blocks=bf16_blocks,
        skip_end_blocks=skip_end_blocks,
        convrot=used_convrot,
        convrot_groupsize=int(convrot_groupsize),
    )


def prepare_projection_maps(
    branch_maps: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    precision: str,
    *,
    skip_end_blocks: int | None = None,
):
    if precision not in PROJECTION_PRECISIONS:
        raise ValueError(f"unknown VDN projection precision {precision!r}")
    if precision == "bf16":
        return tuple(dict(block) for block in branch_maps), None
    if precision == "fp8":
        return prepare_fp8_projection_maps(
            branch_maps,
            device,
            skip_end_blocks=(
                DEFAULT_SKIP_END_BLOCKS if skip_end_blocks is None else int(skip_end_blocks)
            ),
        )
    return prepare_int8_projection_maps(
        branch_maps,
        device,
        skip_end_blocks=0 if skip_end_blocks is None else int(skip_end_blocks),
    )


def projection_out_features(weights: Mapping[str, Any]) -> int:
    if BF16_WEIGHT_KEY in weights:
        return int(weights[BF16_WEIGHT_KEY].shape[0])
    if FP8_WEIGHT_KEY in weights:
        return int(weights[FP8_WEIGHT_KEY].shape[0])
    raise KeyError("VDN branch has no projection weight")


def _project_int8(rows: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError("INT8/ConvRot VDN projection is inference-only")
    quant_ops = _comfy_int8_components()
    convrot = rows.shape[-1] % INT8_CONVROT_GROUP == 0
    return quant_ops.ck.int8_linear(
        rows.contiguous(),
        qdata,
        scale,
        None,
        rows.dtype,
        convrot=convrot,
        convrot_groupsize=INT8_CONVROT_GROUP,
    )


def project(rows: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if BF16_WEIGHT_KEY in weights:
        return F.linear(rows, weights[BF16_WEIGHT_KEY])
    if FP8_WEIGHT_KEY not in weights:
        raise KeyError("VDN projection state has no supported weight representation")
    quantized = weights[FP8_WEIGHT_KEY]
    scale = weights[FP8_SCALE_KEY]
    if quantized.dtype == torch.int8:
        return _project_int8(rows, quantized, scale)
    return project_fp8(rows, weights)


def detect_base_precision(model_patcher: Any) -> str:
    """Infer the native H3 wide-linear precision family without moving weights."""
    try:
        dm = model_patcher.get_model_object("diffusion_model")
        weight = dm.blocks[0].attn.qkv_proj.weight
    except Exception:
        return "bf16"
    layout = str(getattr(weight, "_layout_cls", ""))
    if layout == "TensorWiseINT8Layout":
        return "int8"
    if "FP8" in layout.upper() or getattr(weight, "dtype", None) in {
        torch.float8_e4m3fn,
        getattr(torch, "float8_e5m2", torch.float8_e4m3fn),
    }:
        return "fp8"
    return "bf16"


__all__ = [
    "BF16_WEIGHT_KEY",
    "FP8_SCALE_KEY",
    "FP8_WEIGHT_KEY",
    "FP8ProjectionInfo",
    "INT8_CONVROT_GROUP",
    "INT8_SCALE_KEY",
    "INT8_WEIGHT_KEY",
    "INT8ProjectionInfo",
    "PROJECTION_PRECISIONS",
    "PROJECTION_SCALE_KEYS",
    "QUANTIZED_PROJECTION_KEYS",
    "detect_base_precision",
    "fp8_supported",
    "int8_supported",
    "prepare_int8_projection_maps",
    "prepare_projection_maps",
    "project",
    "projection_out_features",
]
