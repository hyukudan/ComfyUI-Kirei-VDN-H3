"""Optional exact overlap of the VDN linear branch with local softmax.

The low-memory runtime evaluates the linear branch before RoPE so it can reuse native
QKV storage. Large-VRAM workstations can make the opposite trade: copy compact raw
video/text Q/K/V, run the far branch on a dedicated CUDA stream, and let the default
stream execute fused QK preparation, window softmax and H3 output projection
concurrently. The two branches join only at their trained sum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .hybrid import (
    _STATE_PATCH,
    _WRAPPER_KEY,
    _normalise_and_rope,
    _qkv,
    _window_softmax,
    make_layout_wrapper,
    make_vdn_forward as make_serial_vdn_forward,
    validate_h3_model,
)
from .layout import current_layout


_LOG = logging.getLogger("comfy.vdn_h3")


@dataclass
class _ParallelContext:
    stream: torch.cuda.Stream
    raw_ready: torch.cuda.Event
    branch_done: torch.cuda.Event


def _parallel_context(state: Any, device: torch.device) -> _ParallelContext:
    contexts = getattr(state, "_parallel_branch_contexts", None)
    if contexts is None:
        contexts = {}
        state._parallel_branch_contexts = contexts
    key = (device.type, device.index)
    ctx = contexts.get(key)
    if ctx is None:
        with torch.cuda.device(device):
            ctx = _ParallelContext(
                stream=torch.cuda.Stream(device=device),
                raw_ready=torch.cuda.Event(blocking=False),
                branch_done=torch.cuda.Event(blocking=False),
            )
        contexts[key] = ctx
    return ctx


def release_parallel_contexts(state: Any) -> None:
    contexts = getattr(state, "_parallel_branch_contexts", None)
    if not contexts:
        return
    for ctx in contexts.values():
        try:
            ctx.stream.synchronize()
        except Exception:
            pass
    contexts.clear()


def _record_on_stream(tensor: torch.Tensor | None, stream: torch.cuda.Stream) -> None:
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        try:
            tensor.record_stream(stream)
        except Exception:
            pass


def make_parallel_vdn_forward(attn: Any, state: Any, block_index: int):
    """Overlap exact linear/softmax branches on resident large-VRAM execution."""

    serial_forward = make_serial_vdn_forward(attn, state, block_index)
    original_forward = attn.forward
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    config = state.config
    branch = state.branches[block_index]

    def vdn_forward(x, rope_freqs=None, transformer_options=None):
        layout = current_layout(required=False)
        if layout is None:
            return original_forward(
                x, rope_freqs=rope_freqs, transformer_options=transformer_options or {}
            )
        linear_active = bool(config.get("linear_enabled", True)) and not layout.full_cover
        run_inference = state.inference and not torch.is_grad_enabled()
        if (
            not linear_active
            or not run_inference
            or not x.is_cuda
            or state.weight_mode != "resident"
        ):
            return serial_forward(
                x, rope_freqs=rope_freqs, transformer_options=transformer_options
            )
        if int(x.shape[0]) != layout.seq_len:
            raise RuntimeError(
                f"VDN layout has {layout.seq_len} rows but attention received {x.shape[0]}"
            )

        main_stream = torch.cuda.current_stream(x.device)
        ctx = _parallel_context(state, x.device)
        gate_enabled = bool(config.get("enable_softmax_gate", True))
        va, vb = layout.video_start, layout.video_end
        ta, tb = layout.text_start, layout.text_start + layout.text_len
        text_enabled = bool(
            getattr(branch, "enable_text_state", config.get("enable_text_state", False))
        )

        state.prefetch_weights(block_index, x.device, x.dtype, None)
        with state.diagnostics.scope("attention.qkv", x.device):
            q_raw, k_raw, value = _qkv(attn, x)

        # These copies are the deliberate speed-for-memory trade. They preserve the
        # pre-RoPE raw branch inputs while the main stream mutates the native Q/K views.
        with state.diagnostics.scope("parallel.raw_copy", x.device):
            q_video = q_raw[va:vb].clone()
            k_video = k_raw[va:vb].clone()
            v_video = value[va:vb].clone()
            text_k = k_raw[ta:tb].clone() if text_enabled else None
            text_v = value[ta:tb].clone() if text_enabled else None
        ctx.raw_ready.record(main_stream)
        ctx.stream.wait_event(ctx.raw_ready)
        for tensor in (q_video, k_video, v_video, text_k, text_v):
            _record_on_stream(tensor, ctx.stream)

        try:
            with torch.cuda.stream(ctx.stream):
                with state.diagnostics.scope("weights.transfer", x.device):
                    weights = state.weights_on(block_index, x.device, x.dtype, keys=None)
                with state.diagnostics.scope("attention.linear_branch.parallel", x.device):
                    branch_weights, projector = state.projection_view(weights)
                    projected = getattr(branch, "projected_delta", None)
                    if projected is None:
                        readout = branch.readout(
                            branch_weights,
                            x[va:vb],
                            q_video,
                            k_video,
                            v_video,
                            layout.num_frames,
                            layout.tokens_per_frame,
                            layout.bounds,
                            frame_size=layout.frame_size,
                            text_x=x[ta:tb] if text_enabled else None,
                            text_k_raw=text_k,
                            text_v_raw=text_v,
                            skip_ends=(layout.anchor_frames == "both"),
                            inference=True,
                        )
                        linear_delta = projector(readout.to(dtype=x.dtype))
                    else:
                        linear_delta = projected(
                            branch_weights,
                            x[va:vb],
                            q_video,
                            k_video,
                            v_video,
                            layout.num_frames,
                            layout.tokens_per_frame,
                            layout.bounds,
                            frame_size=layout.frame_size,
                            text_x=x[ta:tb] if text_enabled else None,
                            text_k_raw=text_k,
                            text_v_raw=text_v,
                            skip_ends=(layout.anchor_frames == "both"),
                            inference=True,
                            projector=projector,
                        )
                state.mark_weights_consumed(block_index, x.device, x.dtype)
                ctx.branch_done.record(ctx.stream)
        except Exception:
            try:
                ctx.stream.synchronize()
            except Exception:
                pass
            _LOG.exception(
                "VDN-H3 parallel branch setup failed on block %d; using serial path",
                block_index,
            )
            return serial_forward(
                x, rope_freqs=rope_freqs, transformer_options=transformer_options
            )

        _record_on_stream(linear_delta, main_stream)

        with state.diagnostics.scope("attention.rope_norm", x.device):
            query, key = _normalise_and_rope(attn, q_raw, k_raw, rope_freqs)
        with state.diagnostics.scope("attention.softmax", x.device):
            softmax_out, used_backend = _window_softmax(
                query,
                key,
                value,
                layout,
                head_dim**-0.5,
                state.attention_backend,
                transformer_options,
                state.window_cache,
            )
            state.last_attention_backend = used_backend
        if tuple(softmax_out.shape) != (layout.seq_len, heads, head_dim):
            raise RuntimeError(
                f"VDN softmax branch returned {tuple(softmax_out.shape)}; "
                f"expected {(layout.seq_len, heads, head_dim)}"
            )
        del query, key, value, q_raw, k_raw

        if gate_enabled:
            gate = torch.sigmoid(
                F.linear(
                    x,
                    weights["softmax_gate.up.weight"],
                    weights.get("softmax_gate.up.bias"),
                )
            ).reshape(layout.seq_len, heads, 1)
            from .softmax_kernels import apply_softmax_gate
            flat = apply_softmax_gate(softmax_out, gate, state, inference=True)
            del gate, softmax_out
        else:
            flat = softmax_out.reshape(layout.seq_len, -1)
            del softmax_out

        with state.diagnostics.scope("attention.out_proj", x.device):
            out = attn.out_proj(flat.to(dtype=x.dtype))
        del flat

        # Wait only at the actual mathematical dependency. QK prep, softmax, gate and
        # out_proj above are free to overlap with the recurrent branch/projection.
        main_stream.wait_event(ctx.branch_done)
        with state.diagnostics.scope("parallel.join", x.device):
            out[va:vb].add_(linear_delta)
        del linear_delta, q_video, k_video, v_video, text_k, text_v
        return out

    vdn_forward._vdn_h3_forward = True
    vdn_forward._vdn_h3_parallel = True
    vdn_forward._vdn_h3_original = original_forward
    return vdn_forward


def apply_vdn_parallel(model_patcher: Any, state: Any):
    """Install VDN with the large-VRAM two-stream scheduler and full tuned block set."""

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
            f"({collisions[0]})."
        )

    state.attach_weights(model_patcher)
    model_patcher.add_object_patch(_STATE_PATCH, state)
    for index, block in enumerate(blocks):
        model_patcher.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            make_parallel_vdn_forward(block.attn, state, index),
        )
    from .block_kernels import install_block_fusions
    install_block_fusions(model_patcher, state, blocks)
    from .adaln import install_adaln_fp32
    install_adaln_fp32(model_patcher, state, _dm)
    try:
        from comfy.patcher_extension import WrappersMP
    except ImportError:
        wrapper_type = "diffusion_model"
    else:
        wrapper_type = WrappersMP.DIFFUSION_MODEL
    model_patcher.add_wrapper_with_key(
        wrapper_type, _WRAPPER_KEY, make_layout_wrapper(state)
    )
    return model_patcher


__all__ = [
    "apply_vdn_parallel",
    "make_parallel_vdn_forward",
    "release_parallel_contexts",
]
