"""Low-rank forward bypass for VDN-H3 adapters.

Uses ComfyUI's reversible PatcherInjection lifecycle while keeping Q/K/V factors
separate. The base quantized weight is never reconstructed for bypass-capable
modules. MiniMax-H3 MLP fc2 is deliberately excluded because Comfy's fused
``linear_input_act`` consumes that weight without calling fc2.forward.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .auxiliary import create_auxiliary_patcher


_LOG = logging.getLogger("comfy.vdn_h3")
_INJECTION_KEY = "vdn_h3_lora_bypass"
_ADDITIONAL_KEY = "vdn_h3_lora_factors"


@dataclass(frozen=True)
class WeightedFactor:
    patch: Any
    strength: float

    @property
    def scale(self) -> float:
        return float(self.strength) * float(self.patch.scale)


class FrugalLoRABypassAdapter(nn.Module):
    """One factorized ``B(A(x))`` term, with no dense B@A materialization."""

    def __init__(self, up: torch.Tensor, down: torch.Tensor, scale: float):
        super().__init__()
        self.up = nn.Parameter(up.detach().to(device="cpu").contiguous(), requires_grad=False)
        self.down = nn.Parameter(down.detach().to(device="cpu").contiguous(), requires_grad=False)
        self.scale = float(scale)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down
        up = self.up
        if down.device != x.device or down.dtype != x.dtype:
            down = down.to(device=x.device, dtype=x.dtype, non_blocking=down.is_pinned())
        if up.device != x.device or up.dtype != x.dtype:
            up = up.to(device=x.device, dtype=x.dtype, non_blocking=up.is_pinned())
        return F.linear(F.linear(x, down), up) * self.scale


class CompositeQKVBypassAdapter(nn.Module):
    """Low-rank terms sharing one fused down projection per native linear module."""

    def __init__(self, terms: Iterable[tuple[FrugalLoRABypassAdapter, tuple[int, int] | None]]):
        super().__init__()
        terms = list(terms)
        if not terms:
            raise ValueError("composite LoRA bypass needs at least one term")
        input_widths = {int(adapter.down.shape[1]) for adapter, _ in terms}
        if len(input_widths) != 1:
            raise ValueError(f"LoRA terms on one module disagree on input width: {sorted(input_widths)}")

        factor_dtypes = [
            dtype
            for adapter, _ in terms
            for dtype in (adapter.down.dtype, adapter.up.dtype)
        ]
        storage_dtype = reduce(torch.promote_types, factor_dtypes)
        down_parts = []
        self.ups = nn.ParameterList()
        self.ranges: list[tuple[int, int]] = []
        self.offsets: list[tuple[int, int] | None] = []
        self.scales: list[float] = []
        cursor = 0
        for adapter, offset in terms:
            down = adapter.down.detach().to(dtype=storage_dtype, device="cpu").contiguous()
            up = adapter.up.detach().to(dtype=storage_dtype, device="cpu").contiguous()
            rank = int(down.shape[0])
            if up.shape[1] != rank:
                raise ValueError("LoRA up/down ranks disagree inside composite bypass")
            if offset is not None and int(up.shape[0]) != int(offset[1]):
                raise ValueError(
                    f"LoRA output width {up.shape[0]} does not match target slice length {offset[1]}"
                )
            down_parts.append(down)
            self.ups.append(nn.Parameter(up, requires_grad=False))
            self.ranges.append((cursor, cursor + rank))
            self.offsets.append(offset)
            self.scales.append(float(adapter.scale))
            cursor += rank
        self.down = nn.Parameter(torch.cat(down_parts, dim=0), requires_grad=False)

    def apply(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        down = self.down
        if down.device != x.device or down.dtype != x.dtype:
            down = down.to(device=x.device, dtype=x.dtype, non_blocking=down.is_pinned())
        hidden = F.linear(x, down)
        for up_param, (start_rank, stop_rank), offset, scale in zip(
            self.ups, self.ranges, self.offsets, self.scales
        ):
            up = up_param
            if up.device != x.device or up.dtype != x.dtype:
                up = up.to(device=x.device, dtype=x.dtype, non_blocking=up.is_pinned())
            delta = F.linear(hidden[..., start_rank:stop_rank], up) * scale
            delta = delta.to(base_out.dtype)
            if offset is None:
                base_out.add_(delta)
            else:
                start, length = offset
                base_out[..., start : start + length].add_(delta)
        return base_out


class LoRABypassRuntime(nn.Module):
    """All bypass factors for one patched model, visible to ComfyUI accounting."""

    def __init__(self, grouped: dict[str, list[WeightedFactor]]):
        super().__init__()
        self.device = torch.device("cpu")
        self.module_paths: list[str] = []
        self.groups = nn.ModuleList()
        for key in sorted(grouped):
            module_path = key.removesuffix(".weight")
            adapters = []
            for weighted in grouped[key]:
                patch = weighted.patch
                adapters.append(
                    (
                        FrugalLoRABypassAdapter(patch.up, patch.down, weighted.scale),
                        patch.offset,
                    )
                )
            self.module_paths.append(module_path)
            self.groups.append(CompositeQKVBypassAdapter(adapters))

    @property
    def nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def forward(self, *args, **kwargs):
        raise RuntimeError("LoRABypassRuntime is storage-only")


class _BypassHook:
    def __init__(self, module: nn.Module, adapter: CompositeQKVBypassAdapter):
        self.module = module
        self.adapter = adapter
        self.original_forward = None

    def _forward(self, x, *args, **kwargs):
        base_out = self.original_forward(x, *args, **kwargs)
        if not isinstance(base_out, torch.Tensor):
            raise RuntimeError("VDN-H3 LoRA bypass expected a tensor linear output")
        return self.adapter.apply(x, base_out)

    def inject(self):
        if self.original_forward is None:
            self.original_forward = self.module.forward
            self.module.forward = self._forward

    def eject(self):
        if self.original_forward is not None:
            self.module.forward = self.original_forward
            self.original_forward = None


def requires_weight_merge(key: str) -> bool:
    return key.endswith(".mlp.fc2.weight")


def partition_factors(factors: Iterable[WeightedFactor], mode: str):
    if mode not in {"auto", "bypass", "merge"}:
        raise ValueError(f"unknown LoRA mode {mode!r}")
    bypass, merge, curve = [], [], []
    for weighted in factors:
        patch = weighted.patch
        if patch.curve_adaln:
            curve.append(weighted)
        elif mode == "merge" or requires_weight_merge(patch.key):
            merge.append(weighted)
        else:
            bypass.append(weighted)
    return bypass, merge, curve


def install_bypass(model_patcher: Any, factors: Iterable[WeightedFactor]):
    factors = list(factors)
    if not factors:
        return 0, None
    grouped: dict[str, list[WeightedFactor]] = defaultdict(list)
    for weighted in factors:
        grouped[weighted.patch.key].append(weighted)
    runtime = LoRABypassRuntime(grouped)
    try:
        factor_patcher = create_auxiliary_patcher(
            runtime,
            model_patcher,
            size=runtime.nbytes,
            label="VDN LoRA bypass factors",
        )
        setter = getattr(model_patcher, "set_additional_models", None)
        if setter is not None:
            setter(_ADDITIONAL_KEY, [factor_patcher])
    except ImportError:
        pass

    hooks: list[_BypassHook] = []
    for path, adapter in zip(runtime.module_paths, runtime.groups):
        try:
            module = model_patcher.get_model_object(path)
        except Exception as exc:
            raise RuntimeError(f"VDN-H3 bypass target {path!r} does not exist") from exc
        hooks.append(_BypassHook(module, adapter))

    try:
        from comfy.patcher_extension import PatcherInjection
    except ImportError as exc:
        raise RuntimeError("VDN-H3 LoRA bypass requires ComfyUI PatcherInjection") from exc

    def inject_all(_patcher):
        for hook in hooks:
            hook.inject()

    def eject_all(_patcher):
        for hook in reversed(hooks):
            hook.eject()

    model_patcher.set_injections(
        _INJECTION_KEY,
        [PatcherInjection(inject=inject_all, eject=eject_all)],
    )
    _LOG.info("VDN-H3 installed %d low-rank bypass terms across %d modules", len(factors), len(hooks))
    return len(factors), runtime


__all__ = [
    "CompositeQKVBypassAdapter",
    "FrugalLoRABypassAdapter",
    "LoRABypassRuntime",
    "WeightedFactor",
    "install_bypass",
    "partition_factors",
    "requires_weight_merge",
]
