"""ComfyUI-facing Video Delta Attention integration.

Only this module knows the native H3 attention contract.  Mathematical branch and
window implementations stay dependency-light in :mod:`branch` and :mod:`window`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import current_layout, layout_from_payload, publish_layout


_LOG = logging.getLogger("comfy.vdn_h3_private")
_STATE_PATCH = "diffusion_model._vdn_h3_private_state"
_WRAPPER_KEY = "vdn_h3_private.layout"


class ManagedBranchWeights(nn.Module):
    """Per-model branch storage with no process-global CUDA references.

    ``resident`` buffers are registered, so normal ``model.to(...)`` and ComfyUI
    offload move them with the patched diffusion model.  ``stream`` storage remains
    on CPU and returns ephemeral device copies for one block.  Both policies have an
    explicit :meth:`release` operation.
    """

    def __init__(self, weights: Sequence[Mapping[str, torch.Tensor]], mode: str = "stream"):
        super().__init__()
        if mode not in {"stream", "resident"}:
            raise ValueError(f"branch weight mode must be 'stream' or 'resident', got {mode!r}")
        self.mode = mode
        self._closed = False
        self._names: list[dict[str, str]] = []
        self._stream: list[dict[str, torch.Tensor]] = []
        for block_index, block in enumerate(weights):
            if not isinstance(block, Mapping) or not block:
                raise ValueError(f"branch block {block_index} has no weight mapping")
            names: dict[str, str] = {}
            cpu: dict[str, torch.Tensor] = {}
            for tensor_index, (key, tensor) in enumerate(sorted(block.items())):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"branch block {block_index} tensor {key!r} is "
                        f"{type(tensor).__name__}, not torch.Tensor"
                    )
                if mode == "resident":
                    name = f"b{block_index}_t{tensor_index}"
                    self.register_buffer(name, tensor.detach().contiguous(), persistent=False)
                    names[str(key)] = name
                else:
                    cpu[str(key)] = tensor.detach().to(device="cpu").contiguous()
            self._names.append(names)
            self._stream.append(cpu)

    def __len__(self) -> int:
        return len(self._names)

    @property
    def closed(self) -> bool:
        return self._closed

    def weights_on(
        self, block_index: int, device: torch.device | str, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        if self._closed:
            raise RuntimeError("VDN-H3 branch weights were released permanently")
        if not 0 <= block_index < len(self):
            raise IndexError(f"VDN-H3 branch block {block_index} is out of range")
        device = torch.device(device)
        if self.mode == "resident":
            source = {
                key: getattr(self, registered)
                for key, registered in self._names[block_index].items()
            }
        else:
            source = self._stream[block_index]
        # No result is cached: streaming copies die after the block, and resident
        # tensors are already managed buffers.  The full device identity is honored.
        return {
            key: tensor
            if tensor.device == device and tensor.dtype == dtype
            else tensor.to(
                device=device,
                dtype=dtype,
                non_blocking=(device.type == "cuda" and tensor.device.type == "cpu" and tensor.is_pinned()),
            )
            for key, tensor in source.items()
        }

    def release(self) -> "ManagedBranchWeights":
        """Move resident buffers back to CPU; streaming storage is already there."""

        if not self._closed and self.mode == "resident":
            self.to(device="cpu")
        return self

    def close(self) -> None:
        """Drop all tensor references.  The patched model cannot run afterwards."""

        if self._closed:
            return
        self.release()
        self._stream.clear()
        for names in self._names:
            for registered in names.values():
                setattr(self, registered, None)
        self._names.clear()
        self._closed = True


class VDNState(nn.Module):
    """One independent, releasable state object for one patched ModelPatcher."""

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
    ):
        super().__init__()
        if attention_backend not in {"grouped", "flex", "reference"}:
            raise ValueError(f"unsupported attention backend {attention_backend!r}")
        self.name = str(name)
        self.config = dict(config)
        # Alias retained for compatibility with the attributed reference API.
        self.cfg = self.config
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_backend = attention_backend
        self.branches = list(branches)
        try:
            from .window import WindowAttentionCache
        except ImportError:
            self.window_cache = None
        else:
            # Cache ownership follows this one patched model and can therefore be
            # released without perturbing any other queued H3 model.
            self.window_cache = WindowAttentionCache(limit=16)
        maps = []
        for index, branch in enumerate(self.branches):
            if branch is None:
                raise ValueError(f"branch {index} is None")
            mapping = branch if isinstance(branch, Mapping) else getattr(branch, "w", None)
            if not isinstance(mapping, Mapping):
                raise TypeError(f"branch {index} does not expose a weight mapping as .w")
            maps.append(mapping)
        self.weight_store = ManagedBranchWeights(maps, mode=weight_mode)
        self.forwards = 0

    def weights_on(self, index: int, device: torch.device | str, dtype: torch.dtype):
        return self.weight_store.weights_on(index, device, dtype)

    def release(self) -> "VDNState":
        self.weight_store.release()
        if self.window_cache is not None:
            self.window_cache.release()
        for branch in self.branches:
            release = getattr(branch, "release", None)
            if release is not None:
                release()
        return self

    def close(self) -> None:
        self.release()
        self.weight_store.close()
        if self.window_cache is not None:
            self.window_cache.clear()


def make_layout_wrapper(state: VDNState):
    """Build a Comfy diffusion-model wrapper with exception-safe context reset."""

    def wrapper(executor, *args, **kwargs):
        try:
            x = args[0] if args else kwargs["x"]
            context = args[2] if len(args) > 2 else kwargs["context"]
        except (IndexError, KeyError) as exc:
            raise TypeError(
                "VDN-H3 could not locate x/context in the MiniMax-H3 forward call"
            ) from exc
        layout = layout_from_payload(kwargs.get("minimax_payload"), x, context, state.config)
        state.forwards += 1
        with publish_layout(layout):
            return executor(*args, **kwargs)

    wrapper._vdn_h3_private_layout_wrapper = True
    return wrapper


def _qkv(attn: Any, x: torch.Tensor):
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    inner = heads * head_dim
    projected = attn.qkv_proj(x)
    if projected.shape[-1] != inner * 3:
        raise RuntimeError(
            f"H3 qkv_proj returned {projected.shape[-1]} features; expected {inner * 3}"
        )
    q, k, v = projected.split(inner, dim=-1)
    seq = x.shape[0]
    return (
        q.view(seq, heads, head_dim),
        k.view(seq, heads, head_dim),
        v.view(seq, heads, head_dim),
    )


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
    # Do not mutate raw q/k retained for the linear branch.
    if getattr(model_management, "in_training", False):
        q4, k4 = quant_ops.ck.rms_rope_split_half(
            q4, k4, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    else:
        q4, k4 = q4.clone(), k4.clone()
        quant_ops.ck.rms_rope_split_half_(
            q4, k4, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    return q4[0], k4[0]


def _dense_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    transformer_options: dict[str, Any] | None,
) -> torch.Tensor:
    """Dense attention returning one invariant shape: ``[S,H,D]``."""

    seq, heads, dim = query.shape
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    except ImportError:
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            scale=dim**-0.5,
        )
        return attended.squeeze(0).transpose(0, 1)

    q = AttentionTensorContainer(query.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(key.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(value.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(
        q,
        k,
        v,
        heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options or {},
    )
    # Comfy's skip_reshape contract is [B,S,H*D].  Canonicalising here is the
    # crucial full-cover gate fix: [S,H*D] cannot broadcast with [S,H,1].
    if out.numel() != seq * heads * dim:
        raise RuntimeError(
            f"dense attention returned shape {tuple(out.shape)}, expected {seq * heads * dim} values"
        )
    return out.reshape(seq, heads, dim)


def _window_softmax(q, k, v, layout, scale, backend, transformer_options, cache=None):
    if backend == "flex":
        try:
            from .window import window_softmax_flex

            return window_softmax_flex(
                q,
                k,
                v,
                layout.video_start,
                layout.video_end,
                layout.num_frames,
                layout.tokens_per_frame,
                layout.bounds,
                scale,
                anchor_frames=layout.anchor_frames,
                cache=cache,
            )
        except (ImportError, RuntimeError, NotImplementedError) as exc:
            _LOG.warning("VDN-H3 FlexAttention unavailable; using grouped SDPA: %s", exc)
    from .window import window_softmax_grouped

    return window_softmax_grouped(
        q,
        k,
        v,
        layout.video_start,
        layout.video_end,
        layout.num_frames,
        layout.tokens_per_frame,
        layout.bounds,
        scale,
        anchor_frames=layout.anchor_frames,
        transformer_options=transformer_options if backend == "reference" else None,
    )


def make_vdn_forward(attn: Any, state: VDNState, block_index: int):
    """Return a reversible object patch for one native H3 attention forward."""

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
        if int(x.shape[0]) != layout.seq_len:
            raise RuntimeError(
                f"VDN layout has {layout.seq_len} rows but attention received {x.shape[0]}"
            )
        q_raw, k_raw, value = _qkv(attn, x)
        linear_active = bool(config.get("linear_enabled", True)) and not layout.full_cover
        if linear_active:
            va, vb = layout.video_start, layout.video_end
            q_video = q_raw[va:vb].clone()
            k_video = k_raw[va:vb].clone()
            v_video = value[va:vb].clone()
            text_x = text_k = text_v = None
            if bool(getattr(branch, "enable_text_state", config.get("enable_text_state", False))):
                ta, tb = layout.text_start, layout.text_start + layout.text_len
                text_x = x[ta:tb]
                text_k = k_raw[ta:tb].clone()
                text_v = value[ta:tb].clone()

        query, key = _normalise_and_rope(attn, q_raw, k_raw, rope_freqs)
        if layout.full_cover:
            softmax_out = _dense_softmax(query, key, value, transformer_options)
        else:
            softmax_out = _window_softmax(
                query,
                key,
                value,
                layout,
                head_dim**-0.5,
                state.attention_backend,
                transformer_options,
                state.window_cache,
            )
        if tuple(softmax_out.shape) != (layout.seq_len, heads, head_dim):
            raise RuntimeError(
                "VDN softmax branch returned "
                f"{tuple(softmax_out.shape)}; expected {(layout.seq_len, heads, head_dim)}"
            )

        weights = state.weights_on(block_index, x.device, x.dtype)
        if bool(config.get("enable_softmax_gate", True)):
            try:
                gate = torch.sigmoid(
                    F.linear(
                        x,
                        weights["softmax_gate.up.weight"],
                        weights.get("softmax_gate.up.bias"),
                    )
                ).reshape(layout.seq_len, heads, 1)
            except KeyError as exc:
                raise RuntimeError(
                    f"VDN branch block {block_index} is missing softmax gate tensor {exc.args[0]!r}"
                ) from exc
            softmax_out = softmax_out * gate.to(dtype=softmax_out.dtype)
        out = attn.out_proj(softmax_out.reshape(layout.seq_len, -1).to(dtype=x.dtype))

        if linear_active:
            readout = branch.readout(
                weights,
                x[layout.video_start : layout.video_end],
                q_video,
                k_video,
                v_video,
                layout.num_frames,
                layout.tokens_per_frame,
                layout.bounds,
                frame_size=layout.frame_size,
                text_x=text_x,
                text_k_raw=text_k,
                text_v_raw=text_v,
                skip_ends=(layout.anchor_frames == "both"),
            )
            try:
                delta = F.linear(readout.to(dtype=x.dtype), weights["to_out_linear.weight"])
            except KeyError as exc:
                raise RuntimeError(
                    f"VDN branch block {block_index} is missing {exc.args[0]!r}"
                ) from exc
            # The linear branch is defined only for target-video rows.  Other packed
            # modalities remain byte-for-byte on the softmax path.
            out[layout.video_start : layout.video_end].add_(delta)
        return out

    vdn_forward._vdn_h3_private_forward = True
    vdn_forward._vdn_h3_private_original = original_forward
    return vdn_forward


def validate_h3_model(model_patcher: Any, block_count: int | None = None):
    """Return native H3 blocks or raise an actionable compatibility error."""

    try:
        dm = model_patcher.get_model_object("diffusion_model")
        blocks = dm.blocks
        first = blocks[0].attn
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Apply VDN-H3 requires a native ComfyUI MiniMax-H3 MODEL with "
            "diffusion_model.blocks[].attn.qkv_proj"
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
            f"VDN checkpoint has {block_count} blocks but the loaded H3 model has "
            f"{len(blocks)}; checkpoint and base model do not match"
        )
    return dm, blocks


def apply_vdn(model_patcher: Any, state: VDNState):
    """Install only reversible ModelPatcher object/wrapper patches."""

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
            f"({collisions[0]}). Apply VDN before the other attention patch or use "
            "an unpatched MiniMax-H3 MODEL."
        )

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
