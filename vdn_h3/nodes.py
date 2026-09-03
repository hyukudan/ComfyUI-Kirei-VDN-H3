"""ComfyUI nodes for the Kirei VDN-H3 integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from .apply import apply_factor_patches
from .bypass import WeightedFactor, install_bypass, partition_factors
from .hybrid import VDNState, apply_vdn, validate_h3_model
from .kernels import COMPILE_POLICIES, KERNEL_BACKENDS
from .projection import PROJECTION_PRECISIONS, detect_base_precision, prepare_projection_maps
from .weights import BRANCH_MODES, PIN_STRATEGIES, STREAMED_PROJECTION_KEY
from .window import ATTENTION_BACKENDS


_LOG = logging.getLogger("comfy.vdn_h3")
_PLACEHOLDER = "<place an authorized VDN checkpoint under models/vdn>"
_EXECUTION_MODES = ("serial", "parallel")
_PROFILES = (
    "auto",
    "max_speed",
    "workstation_fp8",
    "balanced",
    "low_vram",
    "reference",
    "compat_reference",
    "experimental_fp8",
)


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
        expected.update(
            {"softmax_gate.up.weight": (heads, hidden), "softmax_gate.up.bias": (heads,)}
        )
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


def _make_branches(branch_maps, config, heads, head_dim, runtime):
    from .runtime import OptimizedLinearBranch

    return [
        OptimizedLinearBranch(
            weights,
            heads,
            head_dim,
            delta_rule=config.get("delta_rule", "vdn_solve"),
            bridge=config.get("bridge", "alpha"),
            a_fp32=bool(config.get("a_fp32", True)),
            short_conv=tuple(config.get("short_conv", ())),
            enable_text_state=bool(config.get("enable_text_state", False)),
            kernel_backend=runtime["kernel_backend"],
            compile_policy=runtime["compile_policy"],
            tile_frames=runtime["tile_frames"],
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


def _projection_bytes(branch_maps) -> int:
    return sum(
        block[STREAMED_PROJECTION_KEY].numel() * block[STREAMED_PROJECTION_KEY].element_size()
        for block in branch_maps
        if STREAMED_PROJECTION_KEY in block
    )


def _model_cuda_device(model):
    try:
        device = torch.device(getattr(model, "load_device", "cpu"))
    except Exception:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device


def _gpu_budget(model):
    device = _model_cuda_device(model)
    if device is None:
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        free, _ = torch.cuda.mem_get_info(device)
        return device, int(props.total_memory), int(free)
    except Exception:
        return None


def _auto_branch_mode(model, branch_bytes: int, projection_bytes: int | None = None) -> str:
    budget = _gpu_budget(model)
    if budget is None:
        return "stream"
    _device, total, free = budget
    projection_bytes = int(projection_bytes or branch_bytes * 0.90)
    headroom = max(18 * 1024**3, int(total * 0.22))
    if total >= 48 * 1024**3 and free > branch_bytes * 1.20 + headroom:
        return "resident"
    small_resident = max(0, branch_bytes - projection_bytes)
    if free > small_resident + 6 * 1024**3:
        return "hybrid"
    return "stream"


def _auto_tile_frames(model, branch_mode: str) -> int:
    budget = _gpu_budget(model)
    if budget is None:
        return 5
    _device, total, _free = budget
    if branch_mode == "resident" and total >= 48 * 1024**3:
        return 0
    if total <= 32 * 1024**3:
        return 5
    return 0


def _auto_execution_mode(model, branch_mode: str) -> str:
    """Eligibility helper for the opt-in two-stream workstation experiment."""
    if branch_mode != "resident":
        return "serial"
    budget = _gpu_budget(model)
    if budget is None:
        return "serial"
    _device, total, free = budget
    return "parallel" if total >= 48 * 1024**3 and free >= 28 * 1024**3 else "serial"


def _resolve_runtime(
    model,
    branch_maps,
    *,
    profile,
    branch_mode="auto",
    branch_execution="auto",
    lora_mode="auto",
    attention_backend="auto",
    kernel_backend="auto",
    compile_policy="auto",
    tile_frames=0,
    pin_strategy="auto",
    projection_precision="auto",
    linear_kernels=None,
    legacy_branch_weights=None,
):
    if profile not in _PROFILES:
        raise ValueError(f"unknown VDN profile {profile!r}")
    if branch_mode not in {"auto", *BRANCH_MODES}:
        raise ValueError(f"unknown branch mode {branch_mode!r}")
    if branch_execution not in {"auto", *_EXECUTION_MODES}:
        raise ValueError(f"unknown branch execution mode {branch_execution!r}")
    if lora_mode not in {"auto", "bypass", "merge"}:
        raise ValueError(f"unknown LoRA mode {lora_mode!r}")
    if attention_backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend {attention_backend!r}")
    if pin_strategy not in PIN_STRATEGIES:
        raise ValueError(f"unknown pin strategy {pin_strategy!r}")
    if projection_precision not in {"auto", *PROJECTION_PRECISIONS}:
        raise ValueError(f"unknown projection precision {projection_precision!r}")
    if isinstance(tile_frames, bool) or not isinstance(tile_frames, int) or tile_frames < 0:
        raise ValueError("tile_frames must be a non-negative integer")

    if linear_kernels not in {None, "auto"} and kernel_backend == "auto":
        if linear_kernels == "compile":
            kernel_backend = "auto"
            if compile_policy == "auto":
                compile_policy = "shared"
        else:
            kernel_backend = linear_kernels
    if kernel_backend not in {"auto", *KERNEL_BACKENDS, "compile"}:
        raise ValueError(f"unknown kernel backend {kernel_backend!r}")
    if kernel_backend == "compile":
        kernel_backend = "auto"
        if compile_policy == "auto":
            compile_policy = "shared"
    if compile_policy not in {"auto", *COMPILE_POLICIES}:
        raise ValueError(f"unknown compile policy {compile_policy!r}")

    base_precision = detect_base_precision(model)
    total_bytes = _branch_bytes(branch_maps)
    projection_bytes = _projection_bytes(branch_maps)
    if legacy_branch_weights in BRANCH_MODES and branch_mode == "auto":
        resolved_branch = legacy_branch_weights
    elif branch_mode != "auto":
        resolved_branch = branch_mode
    elif profile in {"max_speed", "workstation_fp8"}:
        resolved_branch = "resident"
    elif profile == "low_vram":
        resolved_branch = "hybrid"
    elif profile in {"reference", "compat_reference"}:
        resolved_branch = "stream"
    else:
        resolved_branch = _auto_branch_mode(model, total_bytes, projection_bytes)

    if branch_execution != "auto":
        resolved_execution = branch_execution
    elif profile == "workstation_fp8":
        # Explicit experimental profile only. The qualified upstream single-GPU tuned
        # scheduler is serial, so auto/max_speed never assume two streams are faster.
        resolved_execution = _auto_execution_mode(model, resolved_branch)
    else:
        resolved_execution = "serial"
    if resolved_execution == "parallel" and resolved_branch != "resident":
        raise ValueError("parallel branch execution requires resident branch weights")

    if lora_mode != "auto":
        resolved_lora = lora_mode
    elif profile in {"reference", "compat_reference", "low_vram"}:
        resolved_lora = "bypass"
    elif profile in {"max_speed", "workstation_fp8"}:
        resolved_lora = "merge"
    elif resolved_branch == "resident" and base_precision in {"int8", "fp8"}:
        # Match the official tuned strategy: patch/requantize once at load time instead
        # of paying the low-rank QKV/O GEMMs in all 50 blocks on every denoising step.
        resolved_lora = "merge"
    else:
        resolved_lora = "bypass"

    if attention_backend != "auto":
        resolved_attention = attention_backend
    elif profile == "reference":
        resolved_attention = "reference"
    elif profile == "compat_reference":
        resolved_attention = "compat"
    else:
        resolved_attention = "auto"

    if kernel_backend != "auto":
        resolved_kernel = kernel_backend
    elif profile in {"reference", "compat_reference"}:
        resolved_kernel = "eager"
    else:
        resolved_kernel = "auto"

    if compile_policy != "auto":
        resolved_compile = compile_policy
    elif profile in {"reference", "compat_reference"}:
        resolved_compile = "off"
    elif profile == "max_speed":
        resolved_compile = "reduce_overhead"
    elif resolved_execution == "parallel":
        # Avoid forcing CUDA graph capture across independent model streams.
        resolved_compile = "shared"
    else:
        resolved_compile = "shared"

    if tile_frames > 0:
        resolved_tile = tile_frames
    elif profile == "low_vram":
        resolved_tile = 5
    elif profile in {"reference", "compat_reference", "max_speed", "workstation_fp8"}:
        resolved_tile = 0
    else:
        resolved_tile = _auto_tile_frames(model, resolved_branch)

    if profile in {"reference", "compat_reference"}:
        if projection_precision in {"fp8", "int8"}:
            raise ValueError("quantized VDN projection cannot be used as a reference profile")
        resolved_projection = "bf16"
    elif projection_precision != "auto":
        resolved_projection = projection_precision
    elif profile in {"workstation_fp8", "experimental_fp8"}:
        resolved_projection = "fp8"
    elif profile == "max_speed":
        resolved_projection = base_precision if base_precision in {"int8", "fp8"} else "fp8"
    elif base_precision in {"int8", "fp8"}:
        # A quantized H3 base should not acquire a new 7168->5376 BF16 GEMM in every block.
        resolved_projection = base_precision
    else:
        resolved_projection = "bf16"

    inference = profile not in {"reference", "compat_reference"}
    return {
        "profile": profile,
        "base_precision": base_precision,
        "branch_mode": resolved_branch,
        "branch_execution": resolved_execution,
        "lora_mode": resolved_lora,
        "attention_backend": resolved_attention,
        "kernel_backend": resolved_kernel,
        "compile_policy": resolved_compile,
        "tile_frames": int(resolved_tile),
        "pin_strategy": pin_strategy,
        "projection_precision": resolved_projection,
        "block_fusion": bool(inference and resolved_branch == "resident" and resolved_compile != "off"),
        "inference": inference,
    }


def _finite_strength(value, name):
    if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def apply_checkpoint(
    model,
    checkpoint_name: str,
    *,
    apply_turbo: bool = True,
    strength: float = 1.0,
    default_strength: float | None = None,
    turbo_strength: float | None = None,
    profile: str = "auto",
    branch_mode: str = "auto",
    branch_execution: str = "auto",
    lora_mode: str = "auto",
    attention_backend: str = "auto",
    kernel_backend: str = "auto",
    compile_policy: str = "auto",
    tile_frames: int = 0,
    pin_strategy: str = "auto",
    projection_precision: str = "auto",
    strict_validation: bool = True,
    diagnostics: bool = False,
    linear_kernels: str | None = None,
    branch_weights: str | None = None,
):
    """Load, validate and apply the complete VDN stack transactionally."""
    if checkpoint_name == _PLACEHOLDER:
        raise FileNotFoundError(
            "No VDN checkpoint was found. Put an authorized, complete stage directory "
            "under ComfyUI/models/vdn and refresh the node list."
        )
    global_strength = _finite_strength(strength, "adapter strength")
    default_strength = (
        global_strength if default_strength is None or default_strength < 0
        else _finite_strength(default_strength, "default adapter strength")
    )
    turbo_strength = (
        global_strength if turbo_strength is None or turbo_strength < 0
        else _finite_strength(turbo_strength, "turbo adapter strength")
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
    if linear_dim != head_dim:
        raise RuntimeError(
            "VDN checkpoint/base geometry mismatch: this checkpoint shares H3 raw Q/K/V "
            f"with the linear branch, so linear_head_dim={linear_dim} must equal the "
            f"loaded H3 attention head_dim={head_dim}. A different state dimension "
            "requires a separately trained VDN checkpoint."
        )
    if strict_validation:
        _branch_shapes(
            branch_maps, hidden=hidden, heads=heads, linear_dim=linear_dim,
            gate=bool(config.get("enable_softmax_gate", True)),
        )

    runtime = _resolve_runtime(
        model,
        branch_maps,
        profile=profile,
        branch_mode=branch_mode,
        branch_execution=branch_execution,
        lora_mode=lora_mode,
        attention_backend=attention_backend,
        kernel_backend=kernel_backend,
        compile_policy=compile_policy,
        tile_frames=tile_frames,
        pin_strategy=pin_strategy,
        projection_precision=projection_precision,
        linear_kernels=linear_kernels,
        legacy_branch_weights=branch_weights,
    )
    branches = _make_branches(branch_maps, config, heads, head_dim, runtime)
    storage_maps = branch_maps
    projection_info = None
    if runtime["projection_precision"] != "bf16":
        device = _model_cuda_device(model)
        try:
            if device is None:
                raise RuntimeError("the loaded H3 model has no CUDA load device")
            # Top-speed and native-INT8 profiles quantize every VDN projection, matching
            # the official tuned rung (skip_end_blocks=0). The explicit experimental
            # FP8 profile keeps the conservative FP8 helper default.
            skip_end = (
                0
                if runtime["projection_precision"] == "int8"
                or profile in {"auto", "max_speed", "workstation_fp8"}
                else None
            )
            storage_maps, projection_info = prepare_projection_maps(
                branch_maps,
                device,
                runtime["projection_precision"],
                skip_end_blocks=skip_end,
            )
        except Exception as exc:
            _LOG.warning(
                "VDN-H3 %s projection unavailable (%s); falling back to BF16 before model construction",
                runtime["projection_precision"],
                exc,
            )
            runtime["projection_precision"] = "bf16"
            storage_maps = branch_maps
            projection_info = None

    state = VDNState(
        checkpoint_name,
        config,
        branches,
        heads,
        head_dim,
        weight_mode=runtime["branch_mode"],
        pin_strategy=runtime["pin_strategy"],
        attention_backend=runtime["attention_backend"],
        weight_maps=storage_maps,
        inference=runtime["inference"],
        kernel_backend=runtime["kernel_backend"],
        compile_policy=runtime["compile_policy"],
        tile_frames=runtime["tile_frames"],
        checkpoint_root=path,
        projection_precision=runtime["projection_precision"],
        projection_info=projection_info,
        block_fusion=runtime["block_fusion"],
        diagnostics=diagnostics,
    )
    state.profile = runtime["profile"]
    state.base_precision = runtime["base_precision"]
    state.branch_execution = runtime["branch_execution"]

    cloned = model.clone()
    try:
        if runtime["branch_execution"] == "parallel":
            from .parallel import apply_vdn_parallel
            apply_vdn_parallel(cloned, state)
        else:
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

    per_adapter_strength = {"default": default_strength, "turbo": turbo_strength}
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
            adapter_strength = per_adapter_strength.get(name, global_strength)
            weighted = [WeightedFactor(patch, adapter_strength) for patch in factors]
            bypass, merge, curve = partition_factors(weighted, runtime["lora_mode"])
            merged = 0
            for term in merge:
                merged += apply_factor_patches(cloned, [term.patch], strength=term.strength)
            bypass_terms.extend(bypass)
            curve_terms.extend((term.patch, term.strength) for term in curve)
            if not (bypass or merge or curve):
                raise RuntimeError(f"VDN adapter {name!r} contained no applicable factors")
            reports.append(
                f"{name}@{adapter_strength:.3g}:bypass={len(bypass)},merge={merged},curve={len(curve)}"
            )

        _, lora_runtime = install_bypass(cloned, bypass_terms)
        state.lora_runtime = lora_runtime
        if curve_terms:
            from .curve import apply_curve_adapters
            apply_curve_adapters(cloned, state, curve_terms)
        state.adapters = {
            "active": list(wanted),
            "strengths": {
                name: float(per_adapter_strength.get(name, global_strength)) for name in wanted
            },
            "lora_mode": runtime["lora_mode"],
            "reports": list(reports),
        }
    except Exception:
        state.close()
        raise

    _LOG.info(
        "VDN-H3 %s applied: profile=%s base=%s branch=%s execution=%s lora=%s attention=%s "
        "kernel=%s compile=%s tile_frames=%d pin=%s projection=%s block_fusion=%s; "
        "%d blocks, %.2f GiB branch; adapters %s",
        checkpoint_name,
        runtime["profile"],
        runtime["base_precision"],
        runtime["branch_mode"],
        runtime["branch_execution"],
        runtime["lora_mode"],
        runtime["attention_backend"],
        runtime["kernel_backend"],
        runtime["compile_policy"],
        runtime["tile_frames"],
        runtime["pin_strategy"],
        runtime["projection_precision"],
        runtime["block_fusion"],
        len(blocks),
        state.weight_store.nbytes / 1024**3,
        ", ".join(reports),
    )
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
                "strength": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
                ),
            },
            "optional": {
                "default_adapter_strength": (
                    "FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, **advanced}
                ),
                "turbo_adapter_strength": (
                    "FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, **advanced}
                ),
                "branch_mode": (["auto", *BRANCH_MODES], {"default": "auto", **advanced}),
                "branch_execution": (["auto", *_EXECUTION_MODES], {"default": "auto", **advanced}),
                "lora_mode": (["auto", "bypass", "merge"], {"default": "auto", **advanced}),
                "attention_backend": (list(ATTENTION_BACKENDS), {"default": "auto", **advanced}),
                "kernel_backend": (list(KERNEL_BACKENDS), {"default": "auto", **advanced}),
                "compile_policy": (["auto", *COMPILE_POLICIES], {"default": "auto", **advanced}),
                "tile_frames": (
                    "INT", {"default": 0, "min": 0, "max": 64, "step": 1, **advanced}
                ),
                "pin_strategy": (list(PIN_STRATEGIES), {"default": "auto", **advanced}),
                "projection_precision": (
                    ["auto", *PROJECTION_PRECISIONS], {"default": "auto", **advanced}
                ),
                "strict_validation": ("BOOLEAN", {"default": True, **advanced}),
                "diagnostics": ("BOOLEAN", {"default": False, **advanced}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/video"
    DESCRIPTION = (
        "Apply VideoDeltaNet H3 with native-precision projection, optimized branch "
        "execution, factorized/merged adapters and calibrated attention dispatch."
    )

    def apply(
        self,
        model,
        vdn_checkpoint,
        profile,
        apply_turbo_adapter,
        strength,
        default_adapter_strength=-1.0,
        turbo_adapter_strength=-1.0,
        branch_mode="auto",
        branch_execution="auto",
        lora_mode="auto",
        attention_backend="auto",
        kernel_backend="auto",
        compile_policy="auto",
        tile_frames=0,
        pin_strategy="auto",
        projection_precision="auto",
        strict_validation=True,
        diagnostics=False,
    ):
        return (
            apply_checkpoint(
                model,
                vdn_checkpoint,
                apply_turbo=apply_turbo_adapter,
                strength=strength,
                default_strength=default_adapter_strength,
                turbo_strength=turbo_adapter_strength,
                profile=profile,
                branch_mode=branch_mode,
                branch_execution=branch_execution,
                lora_mode=lora_mode,
                attention_backend=attention_backend,
                kernel_backend=kernel_backend,
                compile_policy=compile_policy,
                tile_frames=tile_frames,
                pin_strategy=pin_strategy,
                projection_precision=projection_precision,
                strict_validation=strict_validation,
                diagnostics=diagnostics,
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
                "strength": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
                ),
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
