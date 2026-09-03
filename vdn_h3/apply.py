"""Transactional application helpers for VDN-H3 ModelPatcher clones."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .hybrid import VDNState, apply_vdn


@dataclass(frozen=True, slots=True)
class DeltaPatch:
    """An additive weight delta, optionally restricted to one tensor slice."""

    key: str
    delta: torch.Tensor
    # Converter coordinates are (row, column); native ModelPatcher slices are
    # (dimension, start, length). validate_delta_patches canonicalises the former.
    offset: tuple[int, ...] | None = None


def _as_delta(item: Any) -> DeltaPatch:
    if isinstance(item, DeltaPatch):
        return item
    if isinstance(item, Mapping):
        key = item.get("key", item.get("target", item.get("target_key")))
        delta = item.get("delta", item.get("tensor"))
        offset = item.get("offset")
        if offset is None and item.get("length") is not None:
            offset = (
                int(item.get("dimension", item.get("dim", 0))),
                int(item.get("start", 0)),
                int(item["length"]),
            )
    else:
        key = getattr(item, "key", getattr(item, "target", getattr(item, "target_key", None)))
        delta = getattr(item, "delta", getattr(item, "tensor", None))
        offset = getattr(item, "offset", None)
        if offset is None and getattr(item, "length", None) is not None:
            offset = (
                int(getattr(item, "dimension", getattr(item, "dim", 0))),
                int(getattr(item, "start", 0)),
                int(item.length),
            )
    if not isinstance(key, str) or not key:
        raise TypeError(f"adapter patch has no valid target key: {item!r}")
    if not isinstance(delta, torch.Tensor):
        raise TypeError(f"adapter patch {key!r} has no tensor delta")
    if offset is not None:
        try:
            offset = tuple(map(int, offset))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"adapter patch {key!r} has invalid offset {offset!r}") from exc
        if len(offset) not in {2, 3} or min(offset) < 0 or (len(offset) == 3 and offset[2] <= 0):
            raise ValueError(
                f"adapter patch {key!r} offset must be (row,column) or "
                "(dimension,start,positive length), "
                f"got {offset!r}"
            )
    if not key.startswith("diffusion_model."):
        key = "diffusion_model." + key
    return DeltaPatch(key, delta.detach().contiguous(), offset)


def normalise_delta_patches(patches: Any) -> list[DeltaPatch]:
    """Accept the converter's explicit deltas or native ModelPatcher-style maps."""

    if patches is None:
        return []
    if isinstance(patches, Mapping):
        out: list[DeltaPatch] = []
        for raw_key, value in patches.items():
            key, offset = (raw_key, None) if isinstance(raw_key, str) else raw_key[:2]
            if isinstance(value, torch.Tensor):
                delta = value
            elif isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
                delta = value[0]
            elif isinstance(value, Mapping) or hasattr(value, "delta"):
                enriched = dict(value) if isinstance(value, Mapping) else value
                if isinstance(enriched, dict):
                    enriched.setdefault("key", key)
                    enriched.setdefault("offset", offset)
                out.append(_as_delta(enriched))
                continue
            else:
                raise TypeError(f"adapter patch {raw_key!r} has unsupported value {type(value).__name__}")
            out.append(_as_delta({"key": key, "delta": delta, "offset": offset}))
        return out
    if isinstance(patches, Iterable) and not isinstance(patches, (str, bytes)):
        return [_as_delta(item) for item in patches]
    raise TypeError(f"unsupported adapter patch collection {type(patches).__name__}")


def _state_dict(model_patcher: Any) -> Mapping[str, torch.Tensor]:
    if hasattr(model_patcher, "model_state_dict"):
        return model_patcher.model_state_dict()
    model = getattr(model_patcher, "model", None)
    if model is None or not hasattr(model, "state_dict"):
        raise TypeError("MODEL is not a compatible ComfyUI ModelPatcher")
    return model.state_dict()


def validate_delta_patches(model_patcher: Any, patches: Any) -> list[DeltaPatch]:
    """Validate every destination and shape before mutating even the clone."""

    normalised = normalise_delta_patches(patches)
    state = _state_dict(model_patcher)
    canonical: list[DeltaPatch] = []
    seen: set[tuple[str, tuple[int, ...] | None]] = set()
    for raw_patch in normalised:
        patch = raw_patch
        if patch.key not in state:
            raise KeyError(
                f"adapter targets {patch.key!r}, which does not exist in the loaded H3 base"
            )
        full_shape = tuple(state[patch.key].shape)
        if patch.offset is not None and len(patch.offset) == 2:
            if len(full_shape) != 2 or patch.delta.ndim != 2:
                raise ValueError(
                    f"coordinate offset {patch.offset!r} for {patch.key} requires a matrix target/delta"
                )
            narrowed = [
                dim
                for dim, (start, size, full) in enumerate(
                    zip(patch.offset, patch.delta.shape, full_shape)
                )
                if start != 0 or size != full
            ]
            if not narrowed:
                patch = DeltaPatch(patch.key, patch.delta, None)
            elif len(narrowed) == 1:
                dim = narrowed[0]
                other = 1 - dim
                if patch.offset[other] != 0 or patch.delta.shape[other] != full_shape[other]:
                    raise ValueError(
                        f"adapter rectangle at {patch.offset!r} cannot be represented by "
                        "ComfyUI's one-dimensional offset patch"
                    )
                patch = DeltaPatch(
                    patch.key,
                    patch.delta,
                    (dim, patch.offset[dim], patch.delta.shape[dim]),
                )
            else:
                raise ValueError(
                    f"adapter rectangle at {patch.offset!r} narrows multiple dimensions; "
                    "ComfyUI ModelPatcher supports one"
                )
        identity = (patch.key, patch.offset)
        if identity in seen:
            raise ValueError(f"duplicate adapter patch destination {identity!r}")
        seen.add(identity)
        expected = full_shape
        if patch.offset is not None:
            dim, start, length = patch.offset
            if dim >= len(expected) or start + length > expected[dim]:
                raise ValueError(
                    f"adapter slice {patch.offset!r} exceeds {patch.key} shape {expected}"
                )
            sliced = list(expected)
            sliced[dim] = length
            expected = tuple(sliced)
        if tuple(patch.delta.shape) != expected:
            raise ValueError(
                f"adapter delta for {patch.key} has shape {tuple(patch.delta.shape)}, "
                f"expected {expected}" + (f" at offset {patch.offset}" if patch.offset else "")
            )
        canonical.append(patch)
    return canonical


def apply_delta_patches(
    model_patcher: Any,
    patches: Any,
    *,
    strength: float = 1.0,
) -> int:
    """Apply exact additive deltas through ModelPatcher; silently skipped keys fail."""

    if not isinstance(strength, (int, float)) or not torch.isfinite(torch.tensor(float(strength))):
        raise ValueError(f"adapter strength must be finite, got {strength!r}")
    normalised = validate_delta_patches(model_patcher, patches)
    native = {
        patch.key if patch.offset is None else (patch.key, patch.offset): (patch.delta,)
        for patch in normalised
    }
    if not native:
        return 0
    applied = model_patcher.add_patches(native, float(strength))
    if len(applied) != len(native):
        missing = sorted(str(key) for key in set(native) - set(applied))
        raise RuntimeError(
            f"ComfyUI accepted only {len(applied)}/{len(native)} VDN adapter deltas; "
            f"refusing a partial model. Missing: {missing[:5]}"
        )
    return len(applied)


def apply_factor_patches(
    model_patcher: Any,
    patches: Iterable[Any],
    *,
    strength: float = 1.0,
) -> int:
    """Register compact LoRA factors with ComfyUI's native weight patcher."""

    from comfy.weight_adapter import LoRAAdapter

    native = {}
    for patch in patches:
        if getattr(patch, "curve_adaln", False):
            raise ValueError(f"curve AdaLN term {patch.source!r} needs runtime injection")
        rank = int(patch.down.shape[0])
        alpha = float(patch.scale) * rank
        adapter = LoRAAdapter(
            set(),
            (patch.up, patch.down, alpha, None, None, None),
        )
        key = patch.key
        if patch.offset is not None:
            start, length = patch.offset
            key = (patch.key, (0, int(start), int(length)))
        if key in native:
            raise ValueError(f"duplicate compact LoRA target {key!r}")
        native[key] = adapter
    if not native:
        return 0
    applied = model_patcher.add_patches(native, float(strength))
    if len(applied) != len(native):
        raise RuntimeError(
            f"ComfyUI accepted only {len(applied)}/{len(native)} compact LoRA patches"
        )
    return len(applied)


def clone_and_apply(
    model: Any,
    state: VDNState,
    *,
    adapter_patches: Any = None,
    adapter_strength: float = 1.0,
) -> Any:
    """Clone first, then install all reversible attention and weight patches."""

    if not hasattr(model, "clone"):
        raise TypeError("Apply VDN-H3 expected a ComfyUI MODEL/ModelPatcher")
    cloned = model.clone()
    apply_vdn(cloned, state)
    if adapter_patches is not None:
        apply_delta_patches(cloned, adapter_patches, strength=adapter_strength)
    return cloned


__all__ = [
    "DeltaPatch",
    "apply_delta_patches",
    "apply_factor_patches",
    "clone_and_apply",
    "normalise_delta_patches",
    "validate_delta_patches",
]
