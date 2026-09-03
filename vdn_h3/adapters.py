"""Exact VDN/Diffusers -> native Comfy MiniMax-H3 weight adapters.

Q, K and V LoRAs are intentionally *not* fused into a synthetic low-rank pair.
Each pair is multiplied in FP32 and represented as a compact row-offset patch on the
native fused QKV weight.  This avoids the large, mostly-zero block-diagonal ``B`` used
by the reference node while retaining exact per-projection alpha/rank scaling.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


_BLOCK = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
_REFINER = re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$")
_LORA_KEY = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$"
)
_KNOWN_PREFIXES = ("base_model.model.", "diffusion_model.")
_QKV_ORDER = {"q": 0, "k": 1, "v": 2}


@dataclass(frozen=True)
class BranchTarget:
    """Native module target for an upstream module."""

    key: str
    qkv_slice: str | None = None
    swap_swiglu_halves: bool = False


@dataclass(frozen=True)
class DeltaPatch:
    """A dense delta for a rectangular slice of a native parameter.

    ``offset`` is expressed in tensor dimensions (row, column).  Q/K/V therefore
    share one ``*.qkv_proj.weight`` target but occupy three disjoint row ranges.
    """

    key: str
    delta: torch.Tensor
    offset: tuple[int, int]
    source: str
    scale: float

    @property
    def end(self) -> tuple[int, int]:
        return (self.offset[0] + self.delta.shape[0], self.offset[1] + self.delta.shape[1])

    def validate_against(self, shape: Sequence[int]) -> "DeltaPatch":
        target = tuple(int(dim) for dim in shape)
        if len(target) != 2:
            raise ValueError(f"{self.key}: target must be a matrix, got {target}")
        if self.offset[0] < 0 or self.offset[1] < 0:
            raise ValueError(f"{self.key}: negative patch offset {self.offset}")
        if self.end[0] > target[0] or self.end[1] > target[1]:
            raise ValueError(
                f"{self.key}: patch {tuple(self.delta.shape)} at {self.offset} exceeds {target}"
            )
        return self


@dataclass(frozen=True)
class FactorPatch:
    """A compact native LoRA patch; no dense ``B @ A`` tensor is materialized."""

    key: str
    up: torch.Tensor
    down: torch.Tensor
    offset: tuple[int, int] | None
    source: str
    scale: float
    curve_adaln: bool = False

    @property
    def output_shape(self) -> tuple[int, int]:
        return int(self.up.shape[0]), int(self.down.shape[1])


def _strip_known_prefix(module: str) -> str:
    for prefix in _KNOWN_PREFIXES:
        if module.startswith(prefix):
            return module[len(prefix):]
    return module


def map_branch_key(key: str) -> str:
    """Map a learned hybrid-branch tensor; inherited/LoRA tensors are forbidden."""
    if ".attn.orig." in key or ".lora_" in key:
        raise ValueError(f"{key!r} is not a branch-only tensor")
    match = _BLOCK.fullmatch(_strip_known_prefix(key))
    if match is None or not match.group(2).startswith("attn."):
        raise ValueError(f"unsupported VDN branch key {key!r}")
    return f"blocks.{match.group(1)}.{match.group(2)}"


def _attention_target(prefix: str, suffix: str) -> BranchTarget:
    for projection in ("q", "k", "v"):
        if suffix in (f"attn.to_{projection}", f"attn.orig.to_{projection}"):
            return BranchTarget(f"{prefix}.attn.qkv_proj", qkv_slice=projection)
    if suffix in ("attn.to_out.0", "attn.orig.to_out.0"):
        return BranchTarget(f"{prefix}.attn.out_proj")
    raise ValueError(f"unsupported attention target {suffix!r}")


def map_lora_target(module: str) -> BranchTarget:
    """Map one upstream Diffusers/PEFT module to the native Comfy module layout."""
    module = _strip_known_prefix(module)
    if module == "norm_out.linear":
        return BranchTarget("final_layer.adaln_proj.linear")

    refiner = _REFINER.fullmatch(module)
    if refiner:
        prefix, suffix = f"token_refiner.blocks.{refiner.group(1)}", refiner.group(2)
        if suffix.startswith("attn."):
            return _attention_target(prefix, suffix)
        if suffix == "ff.net.0.proj":
            return BranchTarget(f"{prefix}.mlp.fc1", swap_swiglu_halves=True)
        if suffix == "ff.net.2":
            return BranchTarget(f"{prefix}.mlp.fc2")
        if suffix == "adaln_proj.linear":
            return BranchTarget(f"{prefix}.adaln_proj.linear")
        raise ValueError(f"unsupported token-refiner LoRA target {module!r}")

    block = _BLOCK.fullmatch(module)
    if block:
        prefix, suffix = f"blocks.{block.group(1)}", block.group(2)
        if suffix.startswith("attn."):
            return _attention_target(prefix, suffix)
        if suffix == "ff.net.0.proj":
            return BranchTarget(f"{prefix}.mlp.fc1", swap_swiglu_halves=True)
        if suffix == "ff.net.2":
            return BranchTarget(f"{prefix}.mlp.fc2")
        if suffix == "adaln_proj.linear":
            return BranchTarget(f"{prefix}.adaln_proj.linear")
        raise ValueError(f"unsupported transformer LoRA target {module!r}")
    raise ValueError(f"unsupported VDN LoRA target {module!r}")


def _adapter_config(adapter_spec: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if adapter_spec is None:
        return {}
    if not isinstance(adapter_spec, Mapping):
        raise ValueError("adapter spec must be a mapping")
    if "config" in adapter_spec:
        if adapter_spec.get("type") != "lora" or adapter_spec.get("version") != 1:
            raise ValueError("only AdapterSpec lora version 1 is supported")
        cfg = adapter_spec["config"]
    else:
        cfg = adapter_spec
    if not isinstance(cfg, Mapping):
        raise ValueError("adapter config must be a mapping")
    return cfg


def _target_matches(pattern: str, module: str) -> bool:
    if "*" in pattern or "?" in pattern or "[" in pattern:
        return fnmatch.fnmatchcase(module, pattern)
    return module == pattern or module.endswith("." + pattern)


def _pattern_value(patterns: Mapping[str, Any], module: str, default: Any, field: str) -> Any:
    matches = [(pattern, value) for pattern, value in patterns.items() if _target_matches(pattern, module)]
    if not matches:
        return default
    values = {value for _, value in matches}
    if len(values) != 1:
        raise ValueError(f"{module}: ambiguous {field} patterns {matches}")
    # Prefer exact over suffix/glob matches, but equal values are semantically identical.
    return matches[0][1]


def parse_adapter_state(
    state: Mapping[str, torch.Tensor],
    adapter_spec: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Parse and completely inventory a PEFT safetensors state.

    Unknown keys, duplicate sides, missing pairs, incompatible ranks, undeclared
    targets and declared target patterns matching no tensor all fail loudly.
    """
    if not isinstance(state, Mapping) or not state:
        raise ValueError("adapter state must be a non-empty mapping")
    cfg = _adapter_config(adapter_spec)
    parsed: dict[str, dict[str, torch.Tensor]] = {}
    adapter_names: set[str] = set()
    for key, tensor in state.items():
        if not isinstance(key, str):
            raise ValueError("adapter tensor keys must be strings")
        match = _LORA_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"unrecognised adapter tensor {key!r}; no tensor is skipped")
        module = _strip_known_prefix(match.group("module"))
        side, adapter_name = match.group("side"), match.group("adapter")
        if adapter_name is not None:
            adapter_names.add(adapter_name)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{key}: LoRA tensors must be rank-2 torch tensors")
        sides = parsed.setdefault(module, {})
        if side in sides:
            raise ValueError(f"duplicate lora_{side} tensor for {module!r}")
        sides[side] = tensor
    if len(adapter_names) > 1:
        raise ValueError(f"one adapter file contains multiple PEFT names: {sorted(adapter_names)}")
    declared_name = cfg.get("name")
    if declared_name is not None and adapter_names and adapter_names != {declared_name}:
        raise ValueError(f"adapter tensor name {next(iter(adapter_names))!r} != spec name {declared_name!r}")

    rank_default = cfg.get("rank")
    rank_pattern = cfg.get("rank_pattern") or {}
    for module, sides in parsed.items():
        if set(sides) != {"A", "B"}:
            raise ValueError(f"{module}: expected exactly lora_A and lora_B, got {sorted(sides)}")
        a, b = sides["A"], sides["B"]
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"{module}: A rank {a.shape[0]} != B rank {b.shape[1]}")
        if rank_default is not None:
            expected_rank = int(_pattern_value(rank_pattern, module, rank_default, "rank"))
            if a.shape[0] != expected_rank:
                raise ValueError(f"{module}: tensor rank {a.shape[0]} != spec rank {expected_rank}")
        # Mapping here is also validation: unsupported tensors can never disappear later.
        map_lora_target(module)

    targets = cfg.get("targets")
    if targets is not None:
        if not isinstance(targets, list) or not targets:
            raise ValueError("adapter targets must be a non-empty list")
        if cfg.get("exact_targets", False):
            if set(targets) != set(parsed):
                raise ValueError(
                    f"exact adapter inventory mismatch: missing={sorted(set(targets)-set(parsed))}, "
                    f"extra={sorted(set(parsed)-set(targets))}"
                )
        else:
            extra = [module for module in parsed if not any(_target_matches(pattern, module) for pattern in targets)]
            empty = [pattern for pattern in targets if not any(_target_matches(pattern, module) for module in parsed)]
            if extra or empty:
                raise ValueError(f"adapter target inventory mismatch: undeclared={extra}, unmatched_patterns={empty}")
    return parsed


def per_module_scale(adapter_spec: Mapping[str, Any], module: str) -> float:
    """Return exact PEFT alpha/rank scaling for one resolved module."""
    cfg = _adapter_config(adapter_spec)
    if "rank" not in cfg:
        raise ValueError("adapter config is missing rank")
    rank = _pattern_value(cfg.get("rank_pattern") or {}, module, cfg["rank"], "rank")
    alpha = _pattern_value(cfg.get("alpha_pattern") or {}, module, cfg.get("alpha", rank), "alpha")
    if type(rank) is not int or rank <= 0 or type(alpha) not in (int, float) or alpha <= 0:
        raise ValueError(f"{module}: invalid alpha/rank {alpha!r}/{rank!r}")
    return float(alpha) / float(rank)


def _lookup_target_shape(target_shapes: Mapping[str, Sequence[int]] | None, key: str) -> tuple[int, ...] | None:
    if target_shapes is None:
        return None
    value = target_shapes.get(key)
    if value is None and key.endswith(".weight"):
        value = target_shapes.get(key[:-len(".weight")])
    if value is None:
        raise KeyError(f"native model inventory is missing adapter target {key!r}")
    if isinstance(value, torch.Tensor):
        value = value.shape
    return tuple(int(dim) for dim in value)


def convert_adapter(
    state: Mapping[str, torch.Tensor],
    adapter_spec: Mapping[str, Any],
    *,
    target_shapes: Mapping[str, Sequence[int]] | None = None,
    target_prefix: str = "",
) -> tuple[DeltaPatch, ...]:
    """Convert all LoRA pairs into exact, compact native weight-delta patches.

    Deltas are computed in FP32.  For fused QKV, ``offset=(0|q, q|k, 2q|v, 0)`` is
    represented as the corresponding row offset and no block-diagonal factor is ever
    allocated.  ``target_shapes`` makes validation against the live native model
    mandatory at conversion time; callers may omit it only for offline inspection.
    """
    parsed = parse_adapter_state(state, adapter_spec)
    patches: list[DeltaPatch] = []
    occupied: set[tuple[str, int | None]] = set()
    for module in sorted(parsed):
        a, b = parsed[module]["A"], parsed[module]["B"]
        target = map_lora_target(module)
        if target.swap_swiglu_halves:
            if b.shape[0] % 2:
                raise ValueError(f"{module}: SwiGLU lora_B output {b.shape[0]} is not even")
            # Upstream Diffusers packs [value; gate], native H3 packs [gate; value].
            value, gate = b.chunk(2, dim=0)
            b = torch.cat((gate, value), dim=0)
        scale = per_module_scale(adapter_spec, module)
        delta = (b.to(dtype=torch.float32) @ a.to(dtype=torch.float32)).mul_(scale).contiguous()
        key = f"{target_prefix}{target.key}.weight"
        shape = _lookup_target_shape(target_shapes, key)
        if target.qkv_slice is None:
            offset = (0, 0)
            collision = (key, None)
            if collision in occupied:
                raise ValueError(f"multiple adapters map onto the full target {key!r}")
            if shape is not None and tuple(delta.shape) != shape:
                raise ValueError(f"{module}: delta shape {tuple(delta.shape)} != native target {shape}")
        else:
            slice_index = _QKV_ORDER[target.qkv_slice]
            rows = int(delta.shape[0])
            offset = (slice_index * rows, 0)
            collision = (key, slice_index)
            if collision in occupied:
                raise ValueError(f"duplicate {target.qkv_slice.upper()} patch for {key!r}")
            if shape is not None:
                if len(shape) != 2 or shape[0] % 3 or shape[0] // 3 != rows or shape[1] != delta.shape[1]:
                    raise ValueError(
                        f"{module}: QKV slice {tuple(delta.shape)} incompatible with native fused target {shape}"
                    )
        occupied.add(collision)
        patch = DeltaPatch(key, delta, offset, module, scale)
        if shape is not None:
            patch.validate_against(shape)
        patches.append(patch)
    patches.sort(key=lambda patch: (patch.key, patch.offset, patch.source))
    return tuple(patches)


def convert_adapter_factors(
    state: Mapping[str, torch.Tensor],
    adapter_spec: Mapping[str, Any],
    *,
    target_shapes: Mapping[str, Sequence[int]],
    target_prefix: str = "",
) -> tuple[FactorPatch, ...]:
    """Convert an adapter without expanding any low-rank pair.

    A mismatched AdaLN input width is accepted only when its output width matches;
    it denotes ComfyUI's pruned curve representation and is returned as a runtime
    ``curve_adaln`` term. All other mismatches fail before the model is patched.
    """

    parsed = parse_adapter_state(state, adapter_spec)
    patches: list[FactorPatch] = []
    occupied: set[tuple[str, int | None]] = set()
    for module in sorted(parsed):
        down, up = parsed[module]["A"], parsed[module]["B"]
        target = map_lora_target(module)
        if target.swap_swiglu_halves:
            if up.shape[0] % 2:
                raise ValueError(f"{module}: SwiGLU lora_B output {up.shape[0]} is not even")
            value, gate = up.chunk(2, dim=0)
            up = torch.cat((gate, value), dim=0)
        key = f"{target_prefix}{target.key}.weight"
        shape = _lookup_target_shape(target_shapes, key)
        assert shape is not None
        output_shape = (int(up.shape[0]), int(down.shape[1]))
        curve_adaln = False
        offset = None
        slice_index = None
        if target.qkv_slice is not None:
            slice_index = _QKV_ORDER[target.qkv_slice]
            if (
                len(shape) != 2
                or shape[0] % 3
                or shape[0] // 3 != output_shape[0]
                or shape[1] != output_shape[1]
            ):
                raise ValueError(
                    f"{module}: LoRA factors {output_shape} are incompatible with "
                    f"native fused QKV target {shape}"
                )
            offset = (slice_index * output_shape[0], output_shape[0])
        elif tuple(shape) != output_shape:
            if (
                key.endswith(".adaln_proj.linear.weight")
                and len(shape) == 2
                and shape[0] == output_shape[0]
            ):
                curve_adaln = True
            else:
                raise ValueError(
                    f"{module}: LoRA factors produce {output_shape}, native target is {shape}"
                )
        collision = (key, slice_index)
        if collision in occupied:
            raise ValueError(f"duplicate compact LoRA destination {collision!r}")
        occupied.add(collision)
        patches.append(
            FactorPatch(
                key=key,
                up=up.detach().contiguous(),
                down=down.detach().contiguous(),
                offset=offset,
                source=module,
                scale=per_module_scale(adapter_spec, module),
                curve_adaln=curve_adaln,
            )
        )
    return tuple(sorted(patches, key=lambda item: (item.key, item.offset or (-1, -1))))


def patches_by_target(patches: Sequence[DeltaPatch]) -> dict[str, tuple[DeltaPatch, ...]]:
    """Group patches for application while rejecting overlap and shape ambiguity."""
    grouped: dict[str, list[DeltaPatch]] = {}
    for patch in patches:
        grouped.setdefault(patch.key, []).append(patch)
    out: dict[str, tuple[DeltaPatch, ...]] = {}
    for key, items in grouped.items():
        items.sort(key=lambda item: item.offset)
        for left, right in zip(items, items[1:]):
            row_overlap = left.offset[0] < right.end[0] and right.offset[0] < left.end[0]
            col_overlap = left.offset[1] < right.end[1] and right.offset[1] < left.end[1]
            if row_overlap and col_overlap:
                raise ValueError(f"overlapping delta patches for {key}: {left.source}, {right.source}")
        out[key] = tuple(items)
    return out


__all__ = [
    "BranchTarget", "DeltaPatch", "FactorPatch", "convert_adapter",
    "convert_adapter_factors", "map_branch_key",
    "map_lora_target", "parse_adapter_state", "patches_by_target",
    "per_module_scale",
]
