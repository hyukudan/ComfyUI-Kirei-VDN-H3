"""ComfyUI-facing hybrid attention integration for native MiniMax-H3."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .diagnostics import DiagnosticsRecorder
from .layout import current_layout, layout_from_payload, publish_layout
from .runtime import SharedBranchRuntime
from .weights import FP8_STREAMED_PROJECTION_KEY, ManagedBranchWeights, STREAMED_PROJECTION_KEY


_LOG = logging.getLogger("comfy.vdn_h3")
_STATE_PATCH = "diffusion_model._vdn_h3_state"
_WRAPPER_KEY = "vdn_h3.layout"


@dataclass(frozen=True)
class _ShapeOnlyWeight:
    shape: tuple[int, ...]


class VDNState:
    """Independent runtime state for one patched ModelPatcher."""

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        branches: Sequence[Any],
        num_heads: int,
        head_dim: int,
        *,
        weight_mode: str = "stream",
        pin_strategy: str = "auto",
        attention_backend: str = "grouped",
        weight_maps: Sequence[Mapping[str, torch.Tensor]] | None = None,
        inference: bool = False,
        kernel_backend: str = "auto",
        compile_policy: str = "shared",
        tile_frames: int = 0,
        checkpoint_root: str | None = None,
        projection_precision: str = "bf16",
        projection_info: Any = None,
        diagnostics: bool = False,
        linear_kernels: str | None = None,
    ):
        from .fp8 import PROJECTION_PRECISIONS
        from .window import ATTENTION_BACKENDS, WindowAttentionCache

        if linear_kernels is not None:
            if linear_kernels == "eager":
                kernel_backend = "eager"
                compile_policy = "off"
            elif linear_kernels == "compile":
                kernel_backend = "auto"
                compile_policy = "shared"
            else:
                kernel_backend = linear_kernels
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"unsupported attention backend {attention_backend!r}")
        if projection_precision not in PROJECTION_PRECISIONS:
            raise ValueError(f"unsupported projection precision {projection_precision!r}")
        self.name = str(name)
        self.checkpoint_root = checkpoint_root
        self.config = dict(config)
        self.cfg = self.config
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_backend = attention_backend
        self.inference = bool(inference)
        self.kernel_backend = kernel_backend
        self.compile_policy = compile_policy
        self.linear_kernels = kernel_backend
        self.tile_frames = int(tile_frames)
        self.projection_precision = projection_precision
        self.projection_info = projection_info
        self.branches = list(branches)
        self.window_cache = WindowAttentionCache(limit=24)
        self.branch_runtime = SharedBranchRuntime(
            kernel_backend=kernel_backend,
            compile_policy=compile_policy,
            tile_frames=tile_frames,
        )
        self.curve_adapter = None
        self.lora_runtime = None
        self.forwards = 0
        self.last_attention_backend = None
        self.diagnostics = DiagnosticsRecorder(diagnostics)

        if weight_maps is None:
            maps = []
            for index, branch in enumerate(self.branches):
                if branch is None:
                    raise ValueError(f"branch {index} is None")
                mapping = branch if isinstance(branch, Mapping) else getattr(branch, "w", None)
                if not isinstance(mapping, Mapping):
                    raise TypeError(f"branch {index} does not expose a weight mapping as .w")
                maps.append(mapping)
        else:
            maps = list(weight_maps)
            if len(maps) != len(self.branches):
                raise ValueError("weight_maps must match branch count")
        streamed_key = (
            FP8_STREAMED_PROJECTION_KEY
            if projection_precision == "fp8"
            else STREAMED_PROJECTION_KEY
        )
        self.weight_store = ManagedBranchWeights(
            maps,
            mode=weight_mode,
            pin_strategy=pin_strategy,
            streamed_keys=(streamed_key,),
        )
        for branch in self.branches:
            detach = getattr(branch, "detach_weights", None)
            if detach is not None:
                detach()
            set_runtime = getattr(branch, "set_runtime", None)
            if set_runtime is not None:
                set_runtime(
                    runtime_cache=self.branch_runtime,
                    kernel_backend=kernel_backend,
                    compile_policy=compile_policy,
                    tile_frames=tile_frames,
                    diagnostics=self.diagnostics,
                )

    @property
    def weight_mode(self):
        return self.weight_store.mode

    def attach_weights(self, model_patcher):
        return self.weight_store.attach_to(model_patcher)

    def prefetch_weights(self, index, device, dtype, keys: Collection[str] | None = None):
        return self.weight_store.prefetch(index, device, dtype, keys)

    def weights_on(self, index, device, dtype, keys: Collection[str] | None = None):
        return self.weight_store.weights_on(index, device, dtype, keys)

    def mark_weights_consumed(self, index, device, dtype):
        self.weight_store.mark_consumed(index, device, dtype)

    def projection_view(self, weights):
        """Return branch-visible metadata plus the actual projection callable."""
        if self.projection_precision == "bf16":
            return weights, lambda value: F.linear(value, weights[STREAMED_PROJECTION_KEY])
        from .fp8 import FP8_WEIGHT_KEY, project

        if FP8_WEIGHT_KEY not in weights:
            raise RuntimeError("FP8 projection state is missing its quantized weight")
        # OptimizedLinearBranch only needs `.shape` from the canonical key to allocate
        # output/zero anchors; the GEMM itself is supplied by the projector callback.
        branch_weights = dict(weights)
        branch_weights[STREAMED_PROJECTION_KEY] = _ShapeOnlyWeight(
            tuple(int(x) for x in weights[FP8_WEIGHT_KEY].shape)
        )
        return branch_weights, lambda value: project(value, weights)

    def release(self):
        self.weight_store.release()
        self.window_cache.release()
        self.branch_runtime.release()
        for branch in self.branches:
            release = getattr(branch, "release", None)
            if release is not None:
                release()
        if self.curve_adapter is not None:
            release = getattr(self.curve_adapter, "release", None)
            if release is not None:
                release()
        if self.lora_runtime is not None:
            release = getattr(self.lora_runtime, "release", None)
            if release is not None:
                release()
        self.diagnostics.clear()
        return self

    def close(self):
        self.release()
        self.weight_store.close()


def make_layout_wrapper(state: VDNState):
    def wrapper(executor, *args, **kwargs):
        try:
            x = args[0] if args else kwargs["x"]
            context = args[2] if len(args) > 2 else kwargs["context"]
        except (IndexError, KeyError) as exc:
            raise TypeError("VDN-H3 could not locate x/context in the MiniMax-H3 forward call") from exc
        layout = layout_from_payload(kwargs.get("minimax_payload"), x, context, state.config)
        state.forwards += 1
        curve_scope = nullcontext()
        if state.curve_adapter is not None:
            from .curve import curve_runtime_scope
            curve_scope = curve_runtime_scope()
        sample = x[0] if isinstance(x, (tuple, list)) and x else x
        device = getattr(sample, "device", None)
        with publish_layout(layout), curve_scope, state.diagnostics.scope("forward.total", device):
            return executor(*args, **kwargs)

    wrapper._vdn_h3_layout_wrapper = True
    return wrapper


def _qkv(attn: Any, x: torch.Tensor):
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    inner = heads * head_dim
    projected = attn.qkv_proj(x)
    if projected.shape[-1] != inner * 3:
        raise RuntimeError(f"H3 qkv_proj returned {projected.shape[-1]} features; expected {inner * 3}")
    q, k, v = projected.split(inner, dim=-1)
    seq = x.shape[0]
    return q.view(seq, heads, head_dim), k.view(seq, heads, head_dim), v.view(seq, heads, head_dim)


def _normalise_and_rope(attn: Any, q: torch.Tensor, k: torch.Tensor, rope_freqs):
    if rope_freqs is None:
        return attn.q_norm(q), attn.k_norm(k)
    try:
        import comfy.model_management as model_management
        import comfy.quant_ops as quant_ops
    except ImportError as exc:
        raise RuntimeError("RoPE execution requires a compatible ComfyUI runtime") from exc
    seq, heads, dim = q.shape
    q4, k4 = q.view(1, seq, heads, dim), k.view(1, seq, heads, dim)
    qw = model_management.cast_to(attn.q_norm.weight, device=q.device)
    kw = model_management.cast_to(attn.k_norm.weight, device=k.device)
    rot = int(rope_freqs.shape[-3]) * 2
    if getattr(model_management, "in_training", False) or torch.is_grad_enabled():
        q4, k4 = quant_ops.ck.rms_rope_split_half(
            q4, k4, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    else:
        quant_ops.ck.rms_rope_split_half_(
            q4, k4, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    return q4[0], k4[0]


def _dense_softmax(query, key, value, transformer_options, *, exact=False):
    seq, heads, dim = query.shape
    if exact:
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0), scale=dim**-0.5,
        )
        return attended.squeeze(0).transpose(0, 1)
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    except ImportError:
        return _dense_softmax(query, key, value, transformer_options, exact=True)
    q = AttentionTensorContainer(query.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(key.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(value.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(
        q, k, v, heads, mask=None, skip_reshape=True,
        transformer_options=transformer_options or {},
    )
    if out.numel() != seq * heads * dim:
        raise RuntimeError(f"dense attention returned shape {tuple(out.shape)}, expected {seq * heads * dim} values")
    return out.reshape(seq, heads, dim)


def _window_softmax(q, k, v, layout, scale, requested, transformer_options, cache):
    from .window import (
        resolve_attention_backend,
        window_softmax_decomposed,
        window_softmax_flash2,
        window_softmax_flex,
        window_softmax_grouped,
    )

    backend = resolve_attention_backend(
        requested, q, layout.num_frames, layout.bounds, layout.anchor_frames, cache
    )
    if backend == "reference":
        return (
            window_softmax_grouped(
                q, k, v, layout.video_start, layout.video_end, layout.num_frames,
                layout.tokens_per_frame, layout.bounds, scale,
                anchor_frames=layout.anchor_frames, transformer_options=None,
            ),
            backend,
        )
    if backend == "compat":
        return (
            window_softmax_grouped(
                q, k, v, layout.video_start, layout.video_end, layout.num_frames,
                layout.tokens_per_frame, layout.bounds, scale,
                anchor_frames=layout.anchor_frames, transformer_options=transformer_options,
            ),
            backend,
        )
    if backend == "decomposed":
        try:
            return (
                window_softmax_decomposed(
                    q, k, v, layout.video_start, layout.video_end, layout.num_frames,
                    layout.tokens_per_frame, layout.bounds, scale,
                    anchor_frames=layout.anchor_frames, cache=cache,
                ),
                backend,
            )
        except Exception as exc:
            cache.mark_broken("decomposed", exc)
            _LOG.warning("VDN-H3 FA4 attention unavailable; falling back to grouped: %s", exc)
    elif backend == "flash2":
        try:
            return (
                window_softmax_flash2(
                    q, k, v, layout.video_start, layout.video_end, layout.num_frames,
                    layout.tokens_per_frame, layout.bounds, scale,
                    anchor_frames=layout.anchor_frames, cache=cache,
                ),
                backend,
            )
        except Exception as exc:
            cache.mark_broken("flash2", exc)
            _LOG.warning("VDN-H3 FA2 attention unavailable; falling back to grouped: %s", exc)
    elif backend == "flex":
        try:
            return (
                window_softmax_flex(
                    q, k, v, layout.video_start, layout.video_end, layout.num_frames,
                    layout.tokens_per_frame, layout.bounds, scale,
                    anchor_frames=layout.anchor_frames, cache=cache,
                ),
                backend,
            )
        except Exception as exc:
            cache.mark_broken("flex", exc)
            _LOG.warning("VDN-H3 FlexAttention unavailable; falling back to grouped: %s", exc)
    return (
        window_softmax_grouped(
            q, k, v, layout.video_start, layout.video_end, layout.num_frames,
            layout.tokens_per_frame, layout.bounds, scale,
            anchor_frames=layout.anchor_frames, transformer_options=None,
        ),
        "grouped",
    )


def make_vdn_forward(attn: Any, state: VDNState, block_index: int):
    original_forward = attn.forward
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    config = state.config
    branch = state.branches[block_index]

    def vdn_forward(x, rope_freqs=None, transformer_options=None):
        layout = current_layout(required=False)
        if layout is None:
            return original_forward(x, rope_freqs=rope_freqs, transformer_options=transformer_options or {})
        if int(x.shape[0]) != layout.seq_len:
            raise RuntimeError(f"VDN layout has {layout.seq_len} rows but attention received {x.shape[0]}")

        linear_active = bool(config.get("linear_enabled", True)) and not layout.full_cover
        gate_enabled = bool(config.get("enable_softmax_gate", True))
        run_inference = state.inference and not torch.is_grad_enabled()
        if linear_active:
            requested_weights: Collection[str] | None = None
        elif gate_enabled:
            requested_weights = {"softmax_gate.up.weight", "softmax_gate.up.bias"}
        else:
            requested_weights = set()

        if requested_weights is None or requested_weights:
            state.prefetch_weights(block_index, x.device, x.dtype, requested_weights)

        weights_loaded = False
        weights = {}
        linear_delta = None
        deferred_reference = None
        try:
            with state.diagnostics.scope("attention.qkv", x.device):
                q_raw, k_raw, value = _qkv(attn, x)

            if linear_active and run_inference:
                with state.diagnostics.scope("weights.transfer", x.device):
                    weights = state.weights_on(block_index, x.device, x.dtype, keys=None)
                weights_loaded = True
                va, vb = layout.video_start, layout.video_end
                ta, tb = layout.text_start, layout.text_start + layout.text_len
                text_enabled = bool(
                    getattr(branch, "enable_text_state", config.get("enable_text_state", False))
                )
                with state.diagnostics.scope("attention.linear_branch", x.device):
                    projected = getattr(branch, "projected_delta", None)
                    if projected is None:
                        if state.projection_precision != "bf16":
                            raise RuntimeError("FP8 projection requires the optimized VDN branch runtime")
                        readout = branch.readout(
                            weights, x[va:vb], q_raw[va:vb], k_raw[va:vb], value[va:vb],
                            layout.num_frames, layout.tokens_per_frame, layout.bounds,
                            frame_size=layout.frame_size,
                            text_x=x[ta:tb] if text_enabled else None,
                            text_k_raw=k_raw[ta:tb] if text_enabled else None,
                            text_v_raw=value[ta:tb] if text_enabled else None,
                            skip_ends=(layout.anchor_frames == "both"), inference=True,
                        )
                        linear_delta = F.linear(
                            readout.to(dtype=x.dtype), weights[STREAMED_PROJECTION_KEY]
                        )
                    else:
                        branch_weights, projector = state.projection_view(weights)
                        linear_delta = projected(
                            branch_weights, x[va:vb], q_raw[va:vb], k_raw[va:vb], value[va:vb],
                            layout.num_frames, layout.tokens_per_frame, layout.bounds,
                            frame_size=layout.frame_size,
                            text_x=x[ta:tb] if text_enabled else None,
                            text_k_raw=k_raw[ta:tb] if text_enabled else None,
                            text_v_raw=value[ta:tb] if text_enabled else None,
                            skip_ends=(layout.anchor_frames == "both"), inference=True,
                            projector=projector,
                        )
            elif linear_active:
                va, vb = layout.video_start, layout.video_end
                ta, tb = layout.text_start, layout.text_start + layout.text_len
                text_enabled = bool(
                    getattr(branch, "enable_text_state", config.get("enable_text_state", False))
                )
                deferred_reference = (
                    q_raw[va:vb].clone(), k_raw[va:vb].clone(), value[va:vb].clone(),
                    x[ta:tb] if text_enabled else None,
                    k_raw[ta:tb].clone() if text_enabled else None,
                    value[ta:tb].clone() if text_enabled else None,
                )

            with state.diagnostics.scope("attention.rope_norm", x.device):
                query, key = _normalise_and_rope(attn, q_raw, k_raw, rope_freqs)
            exact_dense = state.attention_backend == "reference"
            with state.diagnostics.scope("attention.softmax", x.device):
                if layout.full_cover:
                    softmax_out = _dense_softmax(
                        query, key, value, transformer_options, exact=exact_dense
                    )
                    state.last_attention_backend = "dense_exact" if exact_dense else "dense"
                else:
                    softmax_out, used_backend = _window_softmax(
                        query, key, value, layout, head_dim**-0.5,
                        state.attention_backend, transformer_options, state.window_cache,
                    )
                    state.last_attention_backend = used_backend
            if tuple(softmax_out.shape) != (layout.seq_len, heads, head_dim):
                raise RuntimeError(
                    f"VDN softmax branch returned {tuple(softmax_out.shape)}; "
                    f"expected {(layout.seq_len, heads, head_dim)}"
                )
            del query, key, value, q_raw, k_raw

            if not weights_loaded and (requested_weights is None or requested_weights):
                with state.diagnostics.scope("weights.transfer", x.device):
                    weights = state.weights_on(
                        block_index, x.device, x.dtype, keys=requested_weights
                    )
                weights_loaded = bool(weights)

            if gate_enabled:
                try:
                    gate = torch.sigmoid(
                        F.linear(x, weights["softmax_gate.up.weight"], weights.get("softmax_gate.up.bias"))
                    ).reshape(layout.seq_len, heads, 1)
                except KeyError as exc:
                    raise RuntimeError(
                        f"VDN branch block {block_index} is missing softmax gate tensor {exc.args[0]!r}"
                    ) from exc
                if run_inference:
                    softmax_out.mul_(gate.to(dtype=softmax_out.dtype))
                else:
                    softmax_out = softmax_out * gate.to(dtype=softmax_out.dtype)
                del gate

            with state.diagnostics.scope("attention.out_proj", x.device):
                out = attn.out_proj(softmax_out.reshape(layout.seq_len, -1).to(dtype=x.dtype))
            del softmax_out

            if linear_delta is not None:
                out[layout.video_start : layout.video_end].add_(linear_delta)
                del linear_delta
            elif deferred_reference is not None:
                if not weights_loaded:
                    with state.diagnostics.scope("weights.transfer", x.device):
                        weights = state.weights_on(block_index, x.device, x.dtype, keys=None)
                    weights_loaded = True
                q_video, k_video, v_video, text_x, text_k, text_v = deferred_reference
                with state.diagnostics.scope("attention.linear_branch", x.device):
                    projected = getattr(branch, "projected_delta", None)
                    if projected is None:
                        readout = branch.readout(
                            weights, x[layout.video_start : layout.video_end],
                            q_video, k_video, v_video,
                            layout.num_frames, layout.tokens_per_frame, layout.bounds,
                            frame_size=layout.frame_size,
                            text_x=text_x, text_k_raw=text_k, text_v_raw=text_v,
                            skip_ends=(layout.anchor_frames == "both"), inference=False,
                        )
                        delta = F.linear(readout.to(dtype=x.dtype), weights[STREAMED_PROJECTION_KEY])
                    else:
                        branch_weights, projector = state.projection_view(weights)
                        delta = projected(
                            branch_weights, x[layout.video_start : layout.video_end],
                            q_video, k_video, v_video,
                            layout.num_frames, layout.tokens_per_frame, layout.bounds,
                            frame_size=layout.frame_size,
                            text_x=text_x, text_k_raw=text_k, text_v_raw=text_v,
                            skip_ends=(layout.anchor_frames == "both"), inference=False,
                            projector=projector,
                        )
                    out[layout.video_start : layout.video_end].add_(delta)
                    del delta
            return out
        finally:
            if weights_loaded:
                state.mark_weights_consumed(block_index, x.device, x.dtype)

    vdn_forward._vdn_h3_forward = True
    vdn_forward._vdn_h3_original = original_forward
    return vdn_forward


def validate_h3_model(model_patcher: Any, block_count: int | None = None):
    try:
        dm = model_patcher.get_model_object("diffusion_model")
        blocks = dm.blocks
        first = blocks[0].attn
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Apply VDN-H3 requires a native ComfyUI MiniMax-H3 MODEL with diffusion_model.blocks[].attn.qkv_proj"
        ) from exc
    required = ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads", "head_dim")
    missing = [name for name in required if not hasattr(first, name)]
    if missing:
        raise RuntimeError(
            "loaded MODEL is not a compatible native MiniMax-H3 attention; missing "
            + ", ".join(missing)
        )
    if block_count is not None and len(blocks) != block_count:
        raise RuntimeError(
            f"VDN checkpoint has {block_count} blocks but the loaded H3 model has {len(blocks)}; "
            "checkpoint and base model do not match"
        )
    return dm, blocks


def apply_vdn(model_patcher: Any, state: VDNState):
    _dm, blocks = validate_h3_model(model_patcher, len(state.branches))
    existing = getattr(model_patcher, "object_patches", {})
    if _STATE_PATCH in existing:
        raise RuntimeError("VDN-H3 is already applied to this MODEL; do not chain it twice")
    collisions = [
        f"diffusion_model.blocks.{index}.attn.forward"
        for index in range(len(blocks))
        if f"diffusion_model.blocks.{index}.attn.forward" in existing
    ]
    if collisions:
        raise RuntimeError(
            "VDN-H3 cannot safely replace an existing attention object patch "
            f"({collisions[0]}). Apply VDN before the other attention patch or use an unpatched MiniMax-H3 MODEL."
        )

    state.attach_weights(model_patcher)
    model_patcher.add_object_patch(_STATE_PATCH, state)
    for index, block in enumerate(blocks):
        model_patcher.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            make_vdn_forward(block.attn, state, index),
        )
    try:
        from comfy.patcher_extension import WrappersMP
    except ImportError:
        wrapper_type = "diffusion_model"
    else:
        wrapper_type = WrappersMP.DIFFUSION_MODEL
    model_patcher.add_wrapper_with_key(wrapper_type, _WRAPPER_KEY, make_layout_wrapper(state))
    return model_patcher


__all__ = [
    "ManagedBranchWeights",
    "VDNState",
    "apply_vdn",
    "make_layout_wrapper",
    "make_vdn_forward",
    "validate_h3_model",
]
