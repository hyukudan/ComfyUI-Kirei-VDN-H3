"""Opt-in FP8 execution for the dominant VDN ``to_out_linear`` projection.

This is intentionally never an automatic precision change. The implementation follows
the dispatch facts used by OpenVDN: SM100 uses per-tensor activation/weight scales for
the fast cuBLAS `_scaled_mm` path; pre-SM100 cards use rowwise activation scaling and
per-output-channel weight scales. Only this large projection is quantized here. Gates,
beta, alpha, frame statistics, Cholesky and recurrent state remain BF16/FP32.

Following the upstream accuracy policy, the first and last four transformed blocks stay
in BF16 by default. Those edge blocks contribute disproportionately to FP8 trajectory
error while representing a small share of total projection compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - host dependent
    triton = None
    tl = None


FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_WEIGHT_KEY = "to_out_linear.weight_fp8"
FP8_SCALE_KEY = "to_out_linear.weight_scale"
BF16_WEIGHT_KEY = "to_out_linear.weight"
PROJECTION_PRECISIONS = ("bf16", "fp8")
DEFAULT_SKIP_END_BLOCKS = 4


if triton is not None:  # pragma: no cover - compiled on CUDA hosts

    @triton.jit
    def _quantize_rows_kernel(X, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
        tl.store(S + row, scale)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
            y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
            tl.store(Y + row * K + cols, y.to(Y.dtype.element_ty), mask=cols < K)

    @triton.jit
    def _absmax_kernel(X, OUT, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs, mask=offs < N, other=0.0).to(tl.float32)
        tl.atomic_max(OUT, tl.max(tl.abs(x), axis=0))

    @triton.jit
    def _cast_scaled_kernel(X, Y, S, N, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        scale = tl.load(S)
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@dataclass(frozen=True)
class FP8ProjectionInfo:
    per_tensor: bool
    original_bytes: int
    quantized_bytes: int
    quantized_blocks: int
    bf16_blocks: int
    skip_end_blocks: int


def _per_tensor_for(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 10


def _quantize_weight_cpu(weight: torch.Tensor, *, per_tensor: bool):
    value = weight.detach().to(device="cpu", dtype=torch.float32)
    if per_tensor:
        scale = (value.abs().amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
        q = (value / scale).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
        return q.contiguous(), scale.to(torch.float32).contiguous()
    scale = (value.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    q = (value / scale).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
    return q.contiguous(), scale.reshape(1, -1).to(torch.float32).contiguous()


def _quantize_activation_eager(rows: torch.Tensor, *, per_tensor: bool):
    value = rows.float()
    if per_tensor:
        scale = (value.abs().amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    else:
        scale = (value.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    return (value / scale).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE), scale


def quantize_activation(rows: torch.Tensor, *, per_tensor: bool):
    if rows.ndim != 2 or not rows.is_cuda:
        raise ValueError("FP8 projection activation must be a CUDA [rows, channels] tensor")
    rows = rows.contiguous()
    if triton is None:
        return _quantize_activation_eager(rows, per_tensor=per_tensor)
    if per_tensor:
        n = rows.numel()
        amax = torch.zeros(1, device=rows.device, dtype=torch.float32)
        _absmax_kernel[(triton.cdiv(n, 8192),)](
            rows.view(-1), amax, n, BLOCK=8192, num_warps=8
        )
        scale = (amax / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
        out = torch.empty_like(rows, dtype=FP8_DTYPE)
        _cast_scaled_kernel[(triton.cdiv(n, 8192),)](
            rows.view(-1), out.view(-1), scale, n,
            FP8_MAX=_FP8_MAX, BLOCK=8192, num_warps=8,
        )
        return out, scale
    m, k = rows.shape
    out = torch.empty_like(rows, dtype=FP8_DTYPE)
    scale = torch.empty(m, 1, device=rows.device, dtype=torch.float32)
    _quantize_rows_kernel[(m,)](
        rows, out, scale, k, FP8_MAX=_FP8_MAX, BLOCK_K=1024, num_warps=4
    )
    return out, scale


def fp8_supported(device: torch.device | str, *, probe: bool = True) -> bool:
    try:
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return False
        if not hasattr(torch, "_scaled_mm"):
            return False
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < (8, 9):
            return False
        if not probe:
            return True
        x = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
        w = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
        per_tensor = _per_tensor_for(device)
        xq, xs = _quantize_activation_eager(x, per_tensor=per_tensor)
        if per_tensor:
            ws = (w.abs().amax().float() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
            wq = (w.float() / ws).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
        else:
            raw = (w.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX).clamp_min(1e-12)
            wq = (w.float() / raw).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
            ws = raw.reshape(1, -1)
        torch._scaled_mm(
            xq, wq.t(), scale_a=xs, scale_b=ws,
            out_dtype=torch.bfloat16, use_fast_accum=True,
        )
        return True
    except Exception:
        return False


def prepare_projection_maps(
    branch_maps: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    *,
    skip_end_blocks: int = DEFAULT_SKIP_END_BLOCKS,
):
    """Build heterogeneous BF16-edge / FP8-interior CPU projection masters."""
    device = torch.device(device)
    if not fp8_supported(device, probe=True):
        raise RuntimeError(
            f"FP8 VDN projection is not supported by the active PyTorch/CUDA path on {device}"
        )
    if isinstance(skip_end_blocks, bool) or not isinstance(skip_end_blocks, int) or skip_end_blocks < 0:
        raise ValueError("skip_end_blocks must be a non-negative integer")
    count = len(branch_maps)
    if count == 0:
        raise ValueError("branch_maps cannot be empty")
    if skip_end_blocks * 2 >= count:
        raise ValueError(
            f"skip_end_blocks={skip_end_blocks} leaves no FP8 interior blocks for {count} branches"
        )
    per_tensor = _per_tensor_for(device)
    transformed = []
    original_bytes = quantized_bytes = 0
    quantized_blocks = bf16_blocks = 0
    for index, block in enumerate(branch_maps):
        if BF16_WEIGHT_KEY not in block:
            raise KeyError(f"VDN block {index} has no {BF16_WEIGHT_KEY!r}")
        weight = block[BF16_WEIGHT_KEY]
        original_bytes += int(weight.numel() * weight.element_size())
        keep_bf16 = index < skip_end_blocks or index >= count - skip_end_blocks
        if keep_bf16:
            copied = dict(block)
            transformed.append(copied)
            quantized_bytes += int(weight.numel() * weight.element_size())
            bf16_blocks += 1
            continue
        weight_fp8, scale = _quantize_weight_cpu(weight, per_tensor=per_tensor)
        copied = dict(block)
        copied.pop(BF16_WEIGHT_KEY)
        copied[FP8_WEIGHT_KEY] = weight_fp8
        copied[FP8_SCALE_KEY] = scale
        transformed.append(copied)
        quantized_bytes += int(
            weight_fp8.numel() * weight_fp8.element_size() + scale.numel() * scale.element_size()
        )
        quantized_blocks += 1
    return tuple(transformed), FP8ProjectionInfo(
        per_tensor=per_tensor,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        quantized_blocks=quantized_blocks,
        bf16_blocks=bf16_blocks,
        skip_end_blocks=skip_end_blocks,
    )


def projection_out_features(weights: Mapping[str, torch.Tensor]) -> int:
    if BF16_WEIGHT_KEY in weights:
        return int(weights[BF16_WEIGHT_KEY].shape[0])
    if FP8_WEIGHT_KEY in weights:
        return int(weights[FP8_WEIGHT_KEY].shape[0])
    raise KeyError("VDN branch has no projection weight")


def _dequantized_weight(weight_fp8, weight_scale, dtype):
    if weight_scale.numel() == 1:
        return weight_fp8.to(dtype) * weight_scale.to(dtype)
    return weight_fp8.to(dtype) * weight_scale.t().to(dtype)


def project(rows: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Dispatch BF16 or FP8 projection with a same-quantized-model fallback."""
    if BF16_WEIGHT_KEY in weights:
        return torch.nn.functional.linear(rows, weights[BF16_WEIGHT_KEY])
    weight_fp8 = weights[FP8_WEIGHT_KEY]
    weight_scale = weights[FP8_SCALE_KEY]
    per_tensor = weight_scale.numel() == 1
    # `_scaled_mm` is an inference kernel. If somebody explicitly exercises this
    # experimental projection under autograd, preserve graph semantics by dequantizing
    # the already-quantized weight and using the ordinary Linear spelling.
    if torch.is_grad_enabled():
        return torch.nn.functional.linear(
            rows, _dequantized_weight(weight_fp8, weight_scale, rows.dtype)
        )
    try:
        x_fp8, x_scale = quantize_activation(rows, per_tensor=per_tensor)
        return torch._scaled_mm(
            x_fp8,
            weight_fp8.t(),
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=rows.dtype,
            use_fast_accum=True,
        )
    except Exception:
        return torch.nn.functional.linear(
            rows, _dequantized_weight(weight_fp8, weight_scale, rows.dtype)
        )


__all__ = [
    "BF16_WEIGHT_KEY",
    "DEFAULT_SKIP_END_BLOCKS",
    "FP8ProjectionInfo",
    "FP8_DTYPE",
    "FP8_SCALE_KEY",
    "FP8_WEIGHT_KEY",
    "PROJECTION_PRECISIONS",
    "fp8_supported",
    "prepare_projection_maps",
    "project",
    "projection_out_features",
    "quantize_activation",
]
