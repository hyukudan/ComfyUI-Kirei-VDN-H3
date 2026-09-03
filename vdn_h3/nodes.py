"""ComfyUI nodes for the Kirei VDN-H3 integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from .apply import apply_factor_patches
from .bypass import WeightedFactor, install_bypass, partition_factors
from .hybrid import VDNState, apply_vdn, validate_h3_model


_LOG = logging.getLogger("comfy.vdn_h3")
_PLACEHOLDER = "<place an authorized VDN checkpoint under models/vdn>"
_PROFILES = ("auto", "max_speed", "balanced", "low_vram", "reference")


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


def _branch_shapes(branches, *, hidden, heads, linear_dim, gate):
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
        expected.update({"softmax_gate.up.weight": (heads, hidden), "softmax_gate.up.bias": (heads,)})
    for block_index, weights in enumerate(branches):
        for key, shape in expected.items():
            tensor = weights.get(key)
            if tensor is None:
                raise RuntimeError(f"VDN checkpoint block {block_index} is missing {key!r}")
            if tuple(tensor.shape) != shape:
                raise RuntimeError(
                    f"VDN checkpoint block {block_index} {key} has shape {tuple(tensor.shape)}, "
                    f"expected {shape} for the loaded base"
                )


def _make_branches(branch_maps, config, heads, head_dim, linear_kernels):
    from .branch import LinearBranch
    return [
        LinearBranch(
            weights, heads, head_dim,
            delta_rule=config.get("delta_rule", "vdn_solve"),
            bridge=config.get("bridge", "alpha"),
            a_fp32=bool(config.get("a_fp32", True)),
            short_conv=tuple(config.get("short_conv", ())),
            enable_text_state=bool(config.get("enable_text_state", False)),
            linear_kernels=linear_kernels,
        )
        for weights in branch_maps
    ]


def _adapter_parts(adapter):
    if hasattr(adapter, "state") and hasattr(adapter, "spec"):
        return adapter.state, adapter.spec
    if isinstance(adapter, tuple) and len(adapter) == 2:
        return adapter
    raise TypeError("VDN adapter loader returned an unsupported adapter object")


def _branch_bytes(branch_maps) -> int:
    return sum(t.numel() * t.element_size() for block in branch_maps for t in block.values())


def _model_cuda_device(model):
    device = torch.device(getattr(model, "load_device", "cpu"))
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device


def _auto_branch_mode(model, branch_bytes: int) -> str:
    device = _model_cuda_device(model)
    if device is None:
        return "stream"
    try:
        total = torch.cuda.get_device_properties(device).total_memory
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return "stream"
    headroom = max(16 * 1024**3, int(total * 0.20))
    return "resident" if total >= 48 * 1024**3 and free > branch_bytes * 1.25 + headroom else "stream"


def _resolve_runtime(
    model,
    branch_maps,
    *,
    profile,
    branch_mode,
    lora_mode,
    attention_backend,
    linear_kernels,
    legacy_branch_weights=None,
):
    if profile not in _PROFILES:
        raise ValueError(f"unknown VDN profile {profile!r}")
    if branch_mode not in {"auto", "resident", "stream"}:
        raise ValueError(f"unknown branch mode {branch_mode!r}")
    if lora_mode not in {"auto", "bypass", "merge"}:
        raise ValueError(f"unknown LoRA mode {lora_mode!r}")
    if attention_backend not in {"auto", "grouped", "flex", "decomposed", "reference"}:
        raise ValueError(f"unknown attention backend {attention_backend!r}")
    if linear_kernels not in {"auto", "triton", "compile", "conv1d", "eager"}:
        raise ValueError(f"unknown linear kernel mode {linear_kernels!r}")

    if legacy_branch_weights in {"resident", "stream"} and branch_mode == "auto":
        resolved_branch = legacy_branch_weights
    elif branch_mode != "auto":
        resolved_branch = branch_mode
    elif profile == "max_speed":
        resolved_branch = "resident"
    elif profile in {"low_vram", "reference"}:
        resolved_branch = "stream"
    else:
        resolved_branch = _auto_branch_mode(model, _branch_bytes(branch_maps))

    resolved_lora = lora_mode
    if resolved_lora == "auto":
        resolved_lora = "merge" if profile == "reference" else "bypass"
    resolved_attention = attention_backend
    if resolved_attention == "auto" and profile == "reference":
        resolved_attention = "reference"
    resolved_linear = linear_kernels
    if resolved_linear == "auto" and profile == "reference":
        resolved_linear = "eager"
    return {
        "profile": profile,
        "branch_mode": resolved_branch,
        "lora_mode": resolved_lora,
        "attention_backend": resolved_attention,
        "linear_kernels": resolved_linear,
        "inference": profile != "reference",
    }


def apply_checkpoint(
    model,
    checkpoint_name: str,
    *,
    apply_turbo: bool = True,
    strength: float = 1.0,
    profile: str = "auto",
    branch_mode: str = "auto",
    lora_mode: str = "auto",
    attention_backend: str = "auto",
    linear_kernels: str = "auto",
    strict_validation: bool = True,
    diagnostics: bool = False,
    branch_weights: str | None = None,
):
    """Load, validate and apply the complete VDN stack transactionally."""
    if checkpoint_name == _PLACEHOLDER:
        raise FileNotFoundError(
            "No VDN checkpoint was found. Put an authorized, complete stage directory "
            "under ComfyUI/models/vdn and refresh the node list."
        )
    if not isinstance(strength, (int, float)) or not torch.isfinite(torch.tensor(float(strength))):
        raise ValueError(f"adapter strength must be finite, got {strength!r}")
    from .spec import load_vdn_checkpoint, resolve_vdn_checkpoint

    path = resolve_vdn_checkpoint(checkpoint_name)
    loaded = load_vdn_checkpoint(path)
    config, branch_maps, adapters = _loaded_parts(loaded)
    _dm, blocks = validate_h3_model(model, len(branch_maps))
    first = blocks[0].attn
    heads, head_dim = int(first.heads), int(first.head_dim)
    hidden = int(first.qkv_proj.weight.shape[1])
    linear_dim = int(config.get("linear_head_dim", head_dim))
    if linear_dim != head_dim:
        raise RuntimeError(
            "VDN checkpoint/base geometry mismatch: this native ComfyUI integration "
            f"shares H3 raw Q/K/V with the linear branch, so linear_head_dim={linear_dim} "
            f"must equal the loaded H3 attention head_dim={head_dim}."
        )
    if strict_validation:
        _branch_shapes(
            branch_maps, hidden=hidden, heads=heads, linear_dim=linear_dim,
            gate=bool(config.get("enable_softmax_gate", True)),
        )

    runtime = _resolve_runtime(
        model, branch_maps, profile=profile, branch_mode=branch_mode, lora_mode=lora_mode,
        attention_backend=attention_backend, linear_kernels=linear_kernels,
        legacy_branch_weights=branch_weights,
    )
    branches = _make_branches(branch_maps, config, heads, head_dim, runtime["linear_kernels"])
    state = VDNState(
        checkpoint_name, config, branches, heads, head_dim,
        weight_mode=runtime["branch_mode"], attention_backend=runtime["attention_backend"],
        weight_maps=branch_maps, inference=runtime["inference"],
        linear_kernels=runtime["linear_kernels"], diagnostics=diagnostics,
    )
    cloned = model.clone()
    try:
        apply_vdn(cloned, state)
    except Exception:
        state.close()
        raise

    wanted = ["default"] + (["turbo"] if apply_turbo else [])
    missing = [name for name in wanted if name not in adapters]
    if missing:
        state.close()
        raise RuntimeError(
            f"VDN checkpoint {checkpoint_name!r} is missing required adapter(s) {missing}; refusing a partial model"
        )

    from .adapters import convert_adapter_factors

    target_shapes = {key: tuple(value.shape) for key, value in cloned.model_state_dict().items()}
    bypass_terms: list[WeightedFactor] = []
    curve_terms = []
    reports = []
    try:
        for name in wanted:
            adapter_state, adapter_spec = _adapter_parts(adapters[name])
            factors = convert_adapter_factors(
                adapter_state, adapter_spec,
                target_shapes=target_shapes, target_prefix="diffusion_model.",
            )
            weighted = [WeightedFactor(patch, float(strength)) for patch in factors]
            bypass, merge, curve = partition_factors(weighted, runtime["lora_mode"])
            merged = 0
            for term in merge:
                merged += apply_factor_patches(cloned, [term.patch], strength=term.strength)
            bypass_terms.extend(bypass)
            curve_terms.extend((term.patch, term.strength) for term in curve)
            if not (bypass or merge or curve):
                raise RuntimeError(f"VDN adapter {name!r} contained no applicable factors")
            reports.append(f"{name}:bypass={len(bypass)},merge={merged},curve={len(curve)}")

        _, lora_runtime = install_bypass(cloned, bypass_terms)
        state.lora_runtime = lora_runtime
        if curve_terms:
            from .curve import apply_curve_adapters
            apply_curve_adapters(cloned, state, curve_terms)
    except Exception:
        state.close()
        raise

    _LOG.info(
        "VDN-H3 %s applied: profile=%s branch=%s lora=%s attention=%s linear=%s; "
        "%d blocks, %.2f GiB branch weights; adapters %s",
        checkpoint_name, runtime["profile"], runtime["branch_mode"], runtime["lora_mode"],
        runtime["attention_backend"], runtime["linear_kernels"], len(blocks),
        state.weight_store.nbytes / 1024**3, ", ".join(reports),
    )
    if diagnostics:
        _LOG.info("VDN-H3 diagnostics enabled; synchronized stage timings will be logged during inference")
    return cloned


class KireiApplyVDNH3:
    @classmethod
    def INPUT_TYPES(cls):
        advanced = {"advanced": True}
        return {
            "required": {
                "model": ("MODEL",),
                "vdn_checkpoint": (_checkpoints(),),
                "profile": (list(_PROFILES), {"default": "auto"}),
                "apply_turbo_adapter": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "branch_mode": (["auto", "resident", "stream"], {"default": "auto", **advanced}),
                "lora_mode": (["auto", "bypass", "merge"], {"default": "auto", **advanced}),
                "attention_backend": (
                    ["auto", "grouped", "flex", "decomposed", "reference"],
                    {"default": "auto", **advanced},
                ),
                "linear_kernels": (
                    ["auto", "triton", "compile", "conv1d", "eager"],
                    {"default": "auto", **advanced},
                ),
                "strict_validation": ("BOOLEAN", {"default": True, **advanced}),
                "diagnostics": ("BOOLEAN", {"default": False, **advanced}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/video"
    DESCRIPTION = "Apply VideoDeltaNet H3 with automatic memory, LoRA and attention dispatch."

    def apply(
        self, model, vdn_checkpoint, profile, apply_turbo_adapter, strength,
        branch_mode="auto", lora_mode="auto", attention_backend="auto",
        linear_kernels="auto", strict_validation=True, diagnostics=False,
    ):
        return (
            apply_checkpoint(
                model, vdn_checkpoint, apply_turbo=apply_turbo_adapter, strength=strength,
                profile=profile, branch_mode=branch_mode, lora_mode=lora_mode,
                attention_backend=attention_backend, linear_kernels=linear_kernels,
                strict_validation=strict_validation, diagnostics=diagnostics,
            ),
        )


class KireiApplyVDNH3Alpha:
    """Legacy node id/schema retained so saved workflows still open."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vdn_checkpoint": (_checkpoints(),),
                "apply_turbo_adapter": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "branch_weights": (["stream", "resident"], {"default": "stream"}),
                "attention_backend": (["grouped", "flex", "reference"], {"default": "grouped"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/video/legacy"
    DESCRIPTION = "Legacy Kirei VDN-H3 workflow compatibility node."

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, strength, branch_weights, attention_backend):
        return (
            apply_checkpoint(
                model, vdn_checkpoint, apply_turbo=apply_turbo_adapter, strength=strength,
                profile="balanced", branch_weights=branch_weights,
                attention_backend=attention_backend,
            ),
        )


class KireiReleaseVDNH3Weights:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "release"
    CATEGORY = "model_patches/video/advanced"
    DESCRIPTION = "Explicitly release this model's VDN caches and auxiliary weights."

    def release(self, model):
        state = getattr(model, "object_patches", {}).get("diffusion_model._vdn_h3_state")
        if not isinstance(state, VDNState):
            raise RuntimeError("the supplied MODEL does not carry VDN-H3 state")
        state.release()
        return (model,)


NODE_CLASS_MAPPINGS = {
    "KireiApplyVDNH3": KireiApplyVDNH3,
    "KireiApplyVDNH3Alpha": KireiApplyVDNH3Alpha,
    "KireiReleaseVDNH3Weights": KireiReleaseVDNH3Weights,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiApplyVDNH3": "Kirei Apply VDN-H3",
    "KireiApplyVDNH3Alpha": "Kirei Apply VDN-H3 (Legacy)",
    "KireiReleaseVDNH3Weights": "Kirei Release VDN-H3 Weights",
}


__all__ = [
    "KireiApplyVDNH3",
    "KireiApplyVDNH3Alpha",
    "KireiReleaseVDNH3Weights",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "apply_checkpoint",
]
