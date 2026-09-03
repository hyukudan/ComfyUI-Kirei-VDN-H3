"""ComfyUI-facing hybrid attention integration for native MiniMax-H3."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from .diagnostics import DiagnosticsRecorder
from .layout import current_layout, layout_from_payload, publish_layout
from .weights import ManagedBranchWeights


_LOG = logging.getLogger("comfy.vdn_h3")
_STATE_PATCH = "diffusion_model._vdn_h3_state"
_WRAPPER_KEY = "vdn_h3.layout"


class VDNState:
    """Independent runtime state for one patched ModelPatcher.

    Branch weights live in their own additional ModelPatcher, so attaching this state
    to diffusion_model does not make Comfy move/account the same storage twice.
    """

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        branches: Sequence[Any],
        num_heads: int,
        head_dim: int,
        *,
        weight_mode: str = "stream",
        attention_backend: str = "grouped",
        weight_maps: Sequence[Mapping[str, torch.Tensor]] | None = None,
        inference: bool = False,
        linear_kernels: str = "auto",
        diagnostics: bool = False,
    ):
        from .window import ATTENTION_BACKENDS, WindowAttentionCache

        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"unsupported attention backend {attention_backend!r}")
        self.name = str(name)
        self.config = dict(config)
        self.cfg = self.config
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_backend = attention_backend
        self.inference = bool(inference)
        self.linear_kernels = linear_kernels
        self.branches = list(branches)
        self.window_cache = WindowAttentionCache(limit=16)
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
        self.weight_store = ManagedBranchWeights(maps, mode=weight_mode)
        for branch in self.branches:
            detach = getattr(branch, "detach_weights", None)
            if detach is not None:
                detach()
            set_runtime = getattr(branch, "set_runtime", None)
            if set_runtime is not None:
                set_runtime(linear_kernels=linear_kernels, diagnostics=self.diagnostics)

    @property
    def weight_mode(self):
        return self.weight_store.mode

    def attach_weights(self, model_patcher):
        return self.weight_store.attach_to(model_patcher)

    def weights_on(self, index, device, dtype, keys: Collection[str] | None = None):
        return self.weight_store.weights_on(index, device, dtype, keys)

    def mark_weights_consumed(self, index, device, dtype):
        self.weight_store.mark_consumed(index, device, dtype)

    def release(self):
        self.weight_store.release()
        self.window_cache.release()
        for branch in self.branches:
            release = getattr(branch, "release", None)
            if release is not None:
                release()
        if self.curve_adapter is not None:
            release = getattr(self.curve_adapter, "release", None)
            if release is not None:
                release()
        if self.lora_runtime is not None:
            try:
                self.lora_runtime.to(device="cpu")
            except Exception:
                pass
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
            result = executor(*args, **kwargs)
        if state.diagnostics.enabled and (state.forwards <= 2 or state.forwards % 10 == 0):
            state.diagnostics.log(prefix=f"VDN-H3 {state.name} forward {state.forwards}")
        return result

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


def _dense_softmax(query, key, value, transformer_options):
    seq, heads, dim = query.shape
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    except ImportError:
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0), scale=dim**-0.5,
        )
        return attended.squeeze(0).transpose(0, 1)
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
            _LOG.warning("VDN-H3 decomposed attention unavailable; falling back to grouped: %s", exc)
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
            anchor_frames=layout.anchor_frames,
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

        weights_loaded = False
        try:
            with state.diagnostics.scope("attention.qkv", x.device):
                q_raw, k_raw, value = _qkv(attn, x)
            linear_active = bool(config.get("linear_enabled", True)) and not layout.full_cover
            text_x = text_k = text_v = None
            if linear_active:
                va, vb = layout.video_start, layout.video_end
                q_video = q_raw[va:vb].clone()
                k_video = k_raw[va:vb].clone()
                v_video = value[va:vb].clone()
                if bool(getattr(branch, "enable_text_state", config.get("enable_text_state", False))):
                    ta, tb = layout.text_start, layout.text_start + layout.text_len
                    text_x = x[ta:tb]
                    text_k = k_raw[ta:tb].clone()
                    text_v = value[ta:tb].clone()

            with state.diagnostics.scope("attention.rope_norm", x.device):
                query, key = _normalise_and_rope(attn, q_raw, k_raw, rope_freqs)
            with state.diagnostics.scope("attention.softmax", x.device):
                if layout.full_cover:
                    softmax_out = _dense_softmax(query, key, value, transformer_options)
                    state.last_attention_backend = "dense"
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

            gate_enabled = bool(config.get("enable_softmax_gate", True))
            if linear_active:
                requested_weights = None
            elif gate_enabled:
                requested_weights = {"softmax_gate.up.weight", "softmax_gate.up.bias"}
            else:
                requested_weights = set()
            with state.diagnostics.scope("weights.transfer", x.device):
                weights = (
                    state.weights_on(block_index, x.device, x.dtype, keys=requested_weights or None)
                    if requested_weights or linear_active else {}
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
                if state.inference and not torch.is_grad_enabled():
                    softmax_out.mul_(gate.to(dtype=softmax_out.dtype))
                else:
                    softmax_out = softmax_out * gate.to(dtype=softmax_out.dtype)
                del gate

            with state.diagnostics.scope("attention.out_proj", x.device):
                out = attn.out_proj(softmax_out.reshape(layout.seq_len, -1).to(dtype=x.dtype))
            del softmax_out

            if linear_active:
                run_inference = state.inference and not torch.is_grad_enabled()
                with state.diagnostics.scope("attention.linear_branch", x.device):
                    readout = branch.readout(
                        weights,
                        x[layout.video_start : layout.video_end],
                        q_video, k_video, v_video,
                        layout.num_frames, layout.tokens_per_frame, layout.bounds,
                        frame_size=layout.frame_size,
                        text_x=text_x, text_k_raw=text_k, text_v_raw=text_v,
                        skip_ends=(layout.anchor_frames == "both"),
                        inference=run_inference,
                    )
                    try:
                        delta = F.linear(readout.to(dtype=x.dtype), weights["to_out_linear.weight"])
                    except KeyError as exc:
                        raise RuntimeError(
                            f"VDN branch block {block_index} is missing {exc.args[0]!r}"
                        ) from exc
                    out[layout.video_start : layout.video_end].add_(delta)
                    del readout, delta, q_video, k_video, v_video, text_k, text_v
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
        raise RuntimeError("loaded MODEL is not a compatible native MiniMax-H3 attention; missing " + ", ".join(missing))
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
