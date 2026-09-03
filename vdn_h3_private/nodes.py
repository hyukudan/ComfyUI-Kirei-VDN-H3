"""ComfyUI node definitions.  Heavy Comfy modules are imported only on execution."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from .apply import apply_factor_patches
from .hybrid import VDNState, apply_vdn, validate_h3_model


_LOG = logging.getLogger("comfy.vdn_h3_private")
_PLACEHOLDER = "<place an authorized VDN checkpoint under models/vdn>"


def _checkpoints() -> list[str]:
    try:
        from .spec import list_vdn_checkpoints

        return list_vdn_checkpoints() or [_PLACEHOLDER]
    except (ImportError, OSError, RuntimeError):
        return [_PLACEHOLDER]


def _loaded_parts(loaded):
    if hasattr(loaded, "config") and hasattr(loaded, "branches"):
        return dict(loaded.config), tuple(loaded.branches), dict(loaded.adapters)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        return dict(loaded[0]), tuple(loaded[1]), dict(loaded[2])
    raise TypeError("VDN checkpoint loader returned an unsupported result")


def _branch_shapes(branches, *, hidden: int, heads: int, linear_dim: int, gate: bool):
    expected = {
        "to_out_linear.weight": (hidden, heads * linear_dim),
        "beta_proj.weight": (heads, hidden),
        "norm.weight": (linear_dim,),
        "alpha.A_log": (heads,),
        "alpha.dt_bias": (heads * linear_dim,),
        "alpha.down.weight": (linear_dim, hidden),
        "alpha.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.down.weight": (linear_dim, hidden),
        "output_gate.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.up.bias": (heads * linear_dim,),
    }
    if gate:
        expected.update(
            {
                "softmax_gate.up.weight": (heads, hidden),
                "softmax_gate.up.bias": (heads,),
            }
        )
    for block_index, weights in enumerate(branches):
        for key, shape in expected.items():
            tensor = weights.get(key)
            if tensor is None:
                raise RuntimeError(f"VDN checkpoint block {block_index} is missing {key!r}")
            if tuple(tensor.shape) != shape:
                raise RuntimeError(
                    f"VDN checkpoint block {block_index} {key} has shape "
                    f"{tuple(tensor.shape)}, expected {shape} for the loaded base"
                )


def _make_branches(branch_maps, config, heads, head_dim):
    try:
        from .branch import LinearBranch
    except ImportError as exc:
        raise RuntimeError("VDN-H3 linear branch module is missing from this installation") from exc
    branches = []
    for weights in branch_maps:
        branches.append(
            LinearBranch(
                weights,
                heads,
                head_dim,
                delta_rule=config.get("delta_rule", "vdn_solve"),
                bridge=config.get("bridge", "alpha"),
                a_fp32=bool(config.get("a_fp32", True)),
                short_conv=tuple(config.get("short_conv", ())),
                enable_text_state=bool(config.get("enable_text_state", False)),
            )
        )
    return branches


def _adapter_parts(adapter):
    if hasattr(adapter, "state") and hasattr(adapter, "spec"):
        return adapter.state, adapter.spec
    if isinstance(adapter, tuple) and len(adapter) == 2:
        return adapter
    raise TypeError("VDN adapter loader returned an unsupported adapter object")


def apply_checkpoint(
    model,
    checkpoint_name: str,
    *,
    apply_turbo: bool = True,
    strength: float = 1.0,
    branch_weights: str = "stream",
    attention_backend: str = "grouped",
):
    """Shared implementation, separated from the UI class for synthetic tests."""

    if checkpoint_name == _PLACEHOLDER:
        raise FileNotFoundError(
            "No VDN checkpoint was found. Put an authorized, complete stage directory "
            "under ComfyUI/models/vdn and refresh the node list."
        )
    from .spec import load_vdn_checkpoint, resolve_vdn_checkpoint

    path = resolve_vdn_checkpoint(checkpoint_name)
    loaded = load_vdn_checkpoint(path)
    config, branch_maps, adapters = _loaded_parts(loaded)
    _dm, blocks = validate_h3_model(model, len(branch_maps))
    first = blocks[0].attn
    heads, head_dim = int(first.heads), int(first.head_dim)
    hidden = int(first.qkv_proj.weight.shape[1])
    linear_dim = int(config.get("linear_head_dim", head_dim))
    _branch_shapes(
        branch_maps,
        hidden=hidden,
        heads=heads,
        linear_dim=linear_dim,
        gate=bool(config.get("enable_softmax_gate", True)),
    )
    branches = _make_branches(branch_maps, config, heads, head_dim)
    state = VDNState(
        checkpoint_name,
        config,
        branches,
        heads,
        head_dim,
        weight_mode=branch_weights,
        attention_backend=attention_backend,
    )
    cloned = model.clone()
    apply_vdn(cloned, state)

    wanted = ["default"] + (["turbo"] if apply_turbo else [])
    missing = [name for name in wanted if name not in adapters]
    if missing:
        state.close()
        raise RuntimeError(
            f"VDN checkpoint {checkpoint_name!r} is missing required adapter(s) "
            f"{missing}; refusing a partial model"
        )
    from .adapters import convert_adapter_factors

    target_shapes = {key: tuple(value.shape) for key, value in cloned.model_state_dict().items()}
    reports = []
    curve_terms = []
    try:
        for name in wanted:
            adapter_state, adapter_spec = _adapter_parts(adapters[name])
            factors = convert_adapter_factors(
                adapter_state,
                adapter_spec,
                target_shapes=target_shapes,
                target_prefix="diffusion_model.",
            )
            regular = [patch for patch in factors if not patch.curve_adaln]
            curve = [patch for patch in factors if patch.curve_adaln]
            count = apply_factor_patches(cloned, regular, strength=strength)
            curve_terms.extend((patch, strength) for patch in curve)
            if count + len(curve) == 0:
                raise RuntimeError(f"VDN adapter {name!r} contained no applicable factors")
            reports.append(f"{name}:{count}+{len(curve)}curve")
        if curve_terms:
            from .curve import apply_curve_adapters

            apply_curve_adapters(cloned, state, curve_terms)
    except Exception:
        state.close()
        raise
    _LOG.info(
        "VDN-H3 %s applied to %d blocks (%s; %s weights; adapters %s)",
        checkpoint_name,
        len(blocks),
        attention_backend,
        branch_weights,
        ", ".join(reports),
    )
    return cloned


class KireiApplyVDNH3Alpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vdn_checkpoint": (_checkpoints(),),
                "apply_turbo_adapter": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable the released DMD/turbo adapter when present.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "branch_weights": (
                    ["stream", "resident"],
                    {
                        "default": "stream",
                        "tooltip": "Resident weights are model-managed; stream keeps masters on CPU.",
                    },
                ),
                "attention_backend": (
                    ["grouped", "flex", "reference"],
                    {"default": "grouped"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/video"
    DESCRIPTION = (
        "Apply correctness-first Video Delta Attention to a native MiniMax-H3 model "
        "using reversible ComfyUI ModelPatcher patches."
    )

    def apply(
        self,
        model,
        vdn_checkpoint,
        apply_turbo_adapter,
        strength,
        branch_weights,
        attention_backend,
    ):
        return (
            apply_checkpoint(
                model,
                vdn_checkpoint,
                apply_turbo=apply_turbo_adapter,
                strength=strength,
                branch_weights=branch_weights,
                attention_backend=attention_backend,
            ),
        )


class KireiReleaseVDNH3Weights:
    """Explicitly offload resident VDN weights without global cache surgery."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "release"
    CATEGORY = "model_patches/video"
    DESCRIPTION = "Move this patched model's resident VDN branch weights back to CPU."

    def release(self, model):
        state = getattr(model, "object_patches", {}).get(
            "diffusion_model._vdn_h3_private_state"
        )
        if not isinstance(state, VDNState):
            raise RuntimeError("the supplied MODEL does not carry a VDN-H3 private state")
        state.release()
        return (model,)


NODE_CLASS_MAPPINGS = {
    "KireiApplyVDNH3Alpha": KireiApplyVDNH3Alpha,
    "KireiReleaseVDNH3Weights": KireiReleaseVDNH3Weights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiApplyVDNH3Alpha": "Kirei Apply VideoDeltaNet H3 (Alpha)",
    "KireiReleaseVDNH3Weights": "Kirei Release VDN-H3 Weights",
}


__all__ = [
    "KireiApplyVDNH3Alpha",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "KireiReleaseVDNH3Weights",
    "apply_checkpoint",
]
