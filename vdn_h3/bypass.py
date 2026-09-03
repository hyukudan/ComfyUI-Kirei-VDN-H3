"""Factorized low-rank forward bypass for VDN-H3 adapters.

Q/K/V factors remain independent in output space, but all terms targeting the same
native module share one down GEMM and all terms targeting the same output slice share
one up GEMM. Default+turbo fused QKV therefore evaluates as one down plus Q/K/V ups,
without materializing dense B@A weights.

``mlp.fc2`` is special: native MiniMax-H3 evaluates it through
``comfy.ops.linear_input_act``, which folds the SwiGLU activation into the INT8 kernel
and never calls ``fc2.forward``. Those factors are therefore hooked on the parent MLP:
the native (possibly fused/quantized) base GEMM is kept and the exact low-rank term is
added from the same SwiGLU activation, so the adapter update is never rounded into a
requantized weight.
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

from .auxiliary import create_auxiliary_patcher, unload_auxiliary


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

    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        scale: float,
        *,
        storage_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if storage_dtype is None:
            storage_dtype = torch.promote_types(up.dtype, down.dtype)
        self.up = nn.Parameter(
            up.detach().to(device="cpu", dtype=storage_dtype).contiguous(), requires_grad=False
        )
        self.down = nn.Parameter(
            down.detach().to(device="cpu", dtype=storage_dtype).contiguous(), requires_grad=False
        )
        self.scale = float(scale)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        down, up = self.down, self.up
        if down.device != x.device or down.dtype != x.dtype:
            down = down.to(device=x.device, dtype=x.dtype, non_blocking=down.is_pinned())
        if up.device != x.device or up.dtype != x.dtype:
            up = up.to(device=x.device, dtype=x.dtype, non_blocking=up.is_pinned())
        return F.linear(F.linear(x, down), up) * self.scale


@dataclass(frozen=True)
class _UpGroup:
    offset: tuple[int, int] | None
    start_rank: int
    stop_rank: int
    parameter_index: int


class CompositeQKVBypassAdapter(nn.Module):
    """One down GEMM per native module and one up GEMM per output slice."""

    def __init__(
        self,
        terms: Iterable[tuple[FrugalLoRABypassAdapter, tuple[int, int] | None]],
    ):
        super().__init__()
        terms = list(terms)
        if not terms:
            raise ValueError("composite LoRA bypass needs at least one term")
        input_widths = {int(adapter.down.shape[1]) for adapter, _ in terms}
        if len(input_widths) != 1:
            raise ValueError(
                f"LoRA terms on one module disagree on input width: {sorted(input_widths)}"
            )
        factor_dtypes = [
            dtype for adapter, _ in terms for dtype in (adapter.down.dtype, adapter.up.dtype)
        ]
        storage_dtype = reduce(torch.promote_types, factor_dtypes)

        # Group by output slice and make each group's ranks contiguous in the shared
        # hidden vector. The group's B matrices can then be concatenated column-wise:
        # [B1*s1 | B2*s2] @ [A1(x); A2(x)] == B1A1(x)*s1 + B2A2(x)*s2.
        grouped: dict[tuple[int, int] | None, list[FrugalLoRABypassAdapter]] = defaultdict(list)
        order: list[tuple[int, int] | None] = []
        for adapter, offset in terms:
            if offset not in grouped:
                order.append(offset)
            grouped[offset].append(adapter)

        down_parts = []
        self.group_ups = nn.ParameterList()
        self.groups: list[_UpGroup] = []
        cursor = 0
        for offset in order:
            adapters = grouped[offset]
            group_start = cursor
            up_parts = []
            output_width = None
            for adapter in adapters:
                down = adapter.down.detach().to(dtype=storage_dtype, device="cpu").contiguous()
                up = adapter.up.detach().to(dtype=storage_dtype, device="cpu").contiguous()
                rank = int(down.shape[0])
                if up.shape[1] != rank:
                    raise ValueError("LoRA up/down ranks disagree inside composite bypass")
                if output_width is None:
                    output_width = int(up.shape[0])
                elif int(up.shape[0]) != output_width:
                    raise ValueError("LoRA terms sharing an output slice disagree on output width")
                if offset is not None and int(up.shape[0]) != int(offset[1]):
                    raise ValueError(
                        f"LoRA output width {up.shape[0]} does not match target slice length {offset[1]}"
                    )
                down_parts.append(down)
                # Absorb scalar strength/alpha into B once, outside the hot path.
                up_parts.append(up * float(adapter.scale))
                cursor += rank
            parameter_index = len(self.group_ups)
            self.group_ups.append(
                nn.Parameter(torch.cat(up_parts, dim=1).contiguous(), requires_grad=False)
            )
            self.groups.append(
                _UpGroup(offset, group_start, cursor, parameter_index)
            )
        self.down = nn.Parameter(torch.cat(down_parts, dim=0).contiguous(), requires_grad=False)

    def apply(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        down = self.down
        if down.device != x.device or down.dtype != x.dtype:
            down = down.to(device=x.device, dtype=x.dtype, non_blocking=down.is_pinned())
        hidden = F.linear(x, down)
        for group in self.groups:
            up = self.group_ups[group.parameter_index]
            if up.device != x.device or up.dtype != x.dtype:
                up = up.to(device=x.device, dtype=x.dtype, non_blocking=up.is_pinned())
            delta = F.linear(hidden[..., group.start_rank : group.stop_rank], up).to(base_out.dtype)
            if group.offset is None:
                base_out.add_(delta)
            else:
                start, length = group.offset
                base_out[..., start : start + length].add_(delta)
        return base_out


class LoRABypassRuntime(nn.Module):
    """All bypass factors for one patched model, visible to ComfyUI accounting."""

    def __init__(
        self,
        grouped: dict[str, list[WeightedFactor]],
        *,
        storage_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = torch.device("cpu")
        self.module_paths: list[str] = []
        self.groups = nn.ModuleList()
        self._patcher = None
        for key in sorted(grouped):
            module_path = key.removesuffix(".weight")
            adapters = []
            for weighted in grouped[key]:
                patch = weighted.patch
                adapters.append(
                    (
                        FrugalLoRABypassAdapter(
                            patch.up,
                            patch.down,
                            weighted.scale,
                            storage_dtype=storage_dtype,
                        ),
                        patch.offset,
                    )
                )
            self.module_paths.append(module_path)
            self.groups.append(CompositeQKVBypassAdapter(adapters))

    @property
    def nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def release(self):
        unload_auxiliary(self._patcher, self)
        return self

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


def swiglu(hidden: torch.Tensor) -> torch.Tensor:
    """Native MiniMax-H3 SwiGLU: the first half of ``fc1`` is the gate."""
    gate, up = hidden.chunk(2, dim=-1)
    return F.silu(gate) * up


def _is_int8_quantized(weight) -> bool:
    return getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"


def fc2_base_output(fc2: nn.Module, hidden: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """``fc2(swiglu(hidden))`` through the same path native ComfyUI uses.

    INT8 weights keep the fused ``linear_input_act`` kernel (the activation never reaches
    HBM); every other storage runs the plain projection over the already computed
    activation, which the low-rank term needs anyway.
    """
    if _is_int8_quantized(getattr(fc2, "weight", None)):
        try:
            from comfy.ops import linear_input_act
        except ImportError:
            linear_input_act = None
        if linear_input_act is not None:
            return linear_input_act(fc2, hidden, "swiglu")
    return fc2(activation)


class _MlpBypassHook(_BypassHook):
    """Exact ``mlp.fc2`` adapter term evaluated in activation space on the parent MLP."""

    def _forward(self, x, *args, **kwargs):
        module = self.module
        hidden = module.fc1(x)
        activation = swiglu(hidden)
        base_out = fc2_base_output(module.fc2, hidden, activation)
        if not isinstance(base_out, torch.Tensor):
            raise RuntimeError("VDN-H3 MLP bypass expected a tensor fc2 output")
        return self.adapter.apply(activation, base_out)


_MLP_FC2_SUFFIX = ".mlp.fc2"


def requires_weight_merge(key: str) -> bool:
    """Whether a factor cannot be evaluated in activation space.

    Every supported target now has an exact bypass: ``mlp.fc2`` uses the MLP-level hook
    instead of forcing a weight merge (which requantizes INT8/FP8 weights and rounds
    part of the update away). Kept as an explicit policy point for future targets.
    """
    del key
    return False


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


def _compute_dtype(model_patcher: Any) -> torch.dtype:
    try:
        dtype = model_patcher.model_dtype()
    except Exception:
        dtype = None
    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    # H3 inference activations are normally BF16 even when the base weight storage is
    # INT8/ConvRot. Keeping factors in BF16 avoids a per-forward FP32->BF16 copy.
    return torch.bfloat16


def install_bypass(model_patcher: Any, factors: Iterable[WeightedFactor]):
    factors = list(factors)
    if not factors:
        return 0, None
    grouped: dict[str, list[WeightedFactor]] = defaultdict(list)
    for weighted in factors:
        grouped[weighted.patch.key].append(weighted)
    runtime = LoRABypassRuntime(grouped, storage_dtype=_compute_dtype(model_patcher))
    try:
        factor_patcher = create_auxiliary_patcher(
            runtime,
            model_patcher,
            size=runtime.nbytes,
            label="VDN LoRA bypass factors",
        )
        runtime._patcher = factor_patcher
        setter = getattr(model_patcher, "set_additional_models", None)
        if setter is not None:
            setter(_ADDITIONAL_KEY, [factor_patcher])
    except ImportError:
        pass

    hooks: list[_BypassHook] = []
    for path, adapter in zip(runtime.module_paths, runtime.groups):
        hook_path = path[: -len(".fc2")] if path.endswith(_MLP_FC2_SUFFIX) else path
        try:
            module = model_patcher.get_model_object(hook_path)
        except Exception as exc:
            raise RuntimeError(f"VDN-H3 bypass target {hook_path!r} does not exist") from exc
        if hook_path != path:
            if not hasattr(module, "fc1") or not hasattr(module, "fc2"):
                raise RuntimeError(f"VDN-H3 MLP bypass target {hook_path!r} has no fc1/fc2")
            hooks.append(_MlpBypassHook(module, adapter))
        else:
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
    _LOG.info(
        "VDN-H3 installed %d low-rank bypass terms across %d modules",
        len(factors), len(hooks),
    )
    return len(factors), runtime


__all__ = [
    "CompositeQKVBypassAdapter",
    "FrugalLoRABypassAdapter",
    "LoRABypassRuntime",
    "WeightedFactor",
    "fc2_base_output",
    "install_bypass",
    "partition_factors",
    "requires_weight_merge",
    "swiglu",
]
