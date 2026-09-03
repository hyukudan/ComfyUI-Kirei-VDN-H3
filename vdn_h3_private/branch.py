"""Eager PyTorch implementation of the VDN bidirectional linear branch.

The equations follow the Apache-2.0 OpenVDN reference and the independently
licensed Saganaki22 ComfyUI integration listed in ``THIRD_PARTY.md``.  The code
here is an independent correctness-first implementation: no custom kernels,
checkpoint weights, or live ComfyUI imports are required.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

TEXT_STATE_SCALE = 0.5


class GatherIndexCache:
    """Branch-owned LRU of device tensors used by boundary gathering."""

    def __init__(self, limit: int = 32):
        if limit <= 0:
            raise ValueError("cache limit must be positive")
        self.limit = int(limit)
        self._entries: OrderedDict[
            tuple[Any, ...], dict[str, torch.Tensor]
        ] = OrderedDict()

    def get(self, key: tuple[Any, ...]):
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[Any, ...], value: dict[str, torch.Tensor]) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def release(self) -> None:
        self.clear()

    def __len__(self) -> int:
        return len(self._entries)


class VdnDelta:
    """Solve ``S' = (S diag(alpha) + B) (I + A)^-1`` by Cholesky."""

    def __init__(self, tokens_per_frame: int | None = None):
        del tokens_per_frame

    def factor_apply(
        self, alpha: torch.Tensor, a_raw: torch.Tensor, b_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a_raw.shape != b_raw.shape or a_raw.shape[-1] != a_raw.shape[-2]:
            raise ValueError("A and B must share a square [..., D, D] shape")
        if alpha.shape != a_raw.shape[:-1]:
            raise ValueError("alpha must have shape A.shape[:-1]")
        a32 = a_raw.float()
        b32 = b_raw.float()
        identity = torch.eye(
            a32.shape[-1], device=a32.device, dtype=torch.float32
        ).expand_as(a32)
        chol = torch.linalg.cholesky(a32 + identity)
        inverse_chol = torch.linalg.solve_triangular(
            chol, identity, upper=False, left=True
        )
        inverse = inverse_chol.transpose(-1, -2) @ inverse_chol
        transition = alpha.float().unsqueeze(-1) * inverse
        injection = b32 @ inverse
        return transition.to(a_raw.dtype), injection.to(b_raw.dtype)


class SanaDelta:
    """Scaled first-order delta update used by the alternative Sana rule."""

    def __init__(self, tokens_per_frame: int):
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        self.inv_tokens = 1.0 / tokens_per_frame
        self.inv_sqrt_tokens = self.inv_tokens**0.5

    def factor_apply(
        self, alpha: torch.Tensor, a_raw: torch.Tensor, b_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a_raw.shape != b_raw.shape or a_raw.shape[-1] != a_raw.shape[-2]:
            raise ValueError("A and B must share a square [..., D, D] shape")
        if alpha.shape != a_raw.shape[:-1]:
            raise ValueError("alpha must have shape A.shape[:-1]")
        identity = torch.eye(
            a_raw.shape[-1], device=a_raw.device, dtype=a_raw.dtype
        )
        transition = alpha.unsqueeze(-1) * (identity - self.inv_tokens * a_raw)
        return transition, self.inv_sqrt_tokens * b_raw


DELTA_BACKENDS = {"vdn_solve": VdnDelta, "sana_scaled": SanaDelta}


@contextmanager
def _tf32_matmul(device: torch.device | None = None):
    """Temporarily allow TF32 only when the operation actually runs on CUDA."""
    enabled = device is None or torch.device(device).type == "cuda"
    if not enabled:
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def frame_statistics(
    key_by_frame: torch.Tensor,
    value_by_frame: torch.Tensor,
    beta: torch.Tensor,
    a_fp32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse frame tokens into weighted key/key and value/key matrices.

    Inputs use ``[frames, heads, tokens, dim]`` and beta uses
    ``[frames, heads, tokens]``.  Both returned matrices are fp32 to keep the
    subsequent solve well-conditioned.
    """
    if key_by_frame.ndim != 4 or value_by_frame.shape != key_by_frame.shape:
        raise ValueError("key and value must share shape [frames, heads, tokens, dim]")
    if beta.shape != key_by_frame.shape[:-1]:
        raise ValueError("beta must have shape [frames, heads, tokens]")
    with torch.autocast(device_type=key_by_frame.device.type, enabled=False):
        key32 = key_by_frame.float().contiguous()
        weighted_key32 = (key32 * beta.float().unsqueeze(-1)).contiguous()
        if a_fp32:
            with _tf32_matmul(key_by_frame.device):
                matrix_a = weighted_key32.transpose(-1, -2) @ key32
        else:
            weighted_key = (
                key_by_frame * beta.to(key_by_frame.dtype).unsqueeze(-1)
            ).contiguous()
            matrix_a = (weighted_key.transpose(-1, -2) @ key_by_frame).float()
        # Roundoff can break symmetry enough to bother Cholesky.
        matrix_a = 0.5 * (matrix_a + matrix_a.transpose(-1, -2))
        weighted_value = (
            value_by_frame * beta.to(value_by_frame.dtype).unsqueeze(-1)
        ).contiguous()
        matrix_b = (weighted_value.transpose(-1, -2) @ key_by_frame).float()
    return matrix_a, matrix_b


def run_scans(
    backend: VdnDelta | SanaDelta,
    alpha: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    text_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute inclusive forward and reverse recurrence state banks."""
    if a_raw.ndim != 4 or a_raw.shape != b_raw.shape or a_raw.shape[0] == 0:
        raise ValueError("A and B must share non-empty [frames, heads, D, D] shape")
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, a_raw, b_raw)
        initial = (
            torch.zeros_like(injections[0])
            if text_state is None
            else text_state.to(device=injections.device, dtype=injections.dtype)
        )
        if initial.shape != injections.shape[1:]:
            raise ValueError("text_state must have shape [heads, D, D]")

        forward_states: list[torch.Tensor] = []
        state = initial
        for frame in range(transitions.shape[0]):
            state = torch.baddbmm(injections[frame], state, transitions[frame])
            forward_states.append(state)

        reverse_states: list[torch.Tensor] = [initial] * transitions.shape[0]
        state = initial
        for frame in range(transitions.shape[0] - 1, -1, -1):
            state = torch.baddbmm(injections[frame], state, transitions[frame])
            reverse_states[frame] = state
        return torch.stack(forward_states), torch.stack(reverse_states)


def _device_key(device: torch.device | str) -> tuple[str, int | None]:
    resolved = torch.device(device)
    return resolved.type, resolved.index


def gather_indices(
    bounds: Sequence[tuple[int, int]],
    num_frames: int,
    device: torch.device | str,
    cache: GatherIndexCache | None = None,
) -> dict[str, torch.Tensor]:
    """Return cached recurrence boundary indices with multi-GPU-safe LRU keys."""
    if num_frames <= 0 or len(bounds) != num_frames:
        raise ValueError("bounds must contain one entry for every positive frame count")
    normalized = tuple((int(lo), int(hi)) for lo, hi in bounds)
    for frame, (lo, hi) in enumerate(normalized):
        if lo > frame or hi < frame or lo > hi:
            raise ValueError(f"bounds[{frame}] does not contain its query frame")
    key = (normalized, num_frames, _device_key(device))
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return cached

    last_before = torch.tensor(
        [lo for lo, _ in normalized], dtype=torch.long, device=device
    ) - 1
    first_after = torch.tensor(
        [hi for _, hi in normalized], dtype=torch.long, device=device
    ) + 1
    cached = {
        "before_idx": last_before.clamp(0, num_frames - 1),
        "after_idx": first_after.clamp(0, num_frames - 1),
        "has_before": last_before >= 0,
        "has_after": first_after < num_frames,
        "bridge_before": (last_before + 1).clamp(0, num_frames),
        "bridge_after": first_after.clamp(0, num_frames),
        "frames": torch.arange(num_frames, dtype=torch.long, device=device),
    }
    if cache is not None:
        cache.put(key, cached)
    return cached


def gather_linear_state(
    prefix_states: torch.Tensor,
    suffix_states: torch.Tensor,
    alpha: torch.Tensor,
    bounds: Sequence[tuple[int, int]],
    bridge: str = "alpha",
    text_state: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    cache: GatherIndexCache | None = None,
) -> torch.Tensor:
    """Gather the recurrent complement outside each local softmax window."""
    if bridge not in ("alpha", "none"):
        raise ValueError("bridge must be 'alpha' or 'none'")
    if prefix_states.ndim != 4 or prefix_states.shape != suffix_states.shape:
        raise ValueError("prefix and suffix must share [frames, heads, D, D] shape")
    num_frames, heads, value_dim, key_dim = prefix_states.shape
    if alpha.shape != (num_frames, heads, key_dim):
        raise ValueError("alpha has incompatible [frames, heads, key_dim] shape")
    indices = gather_indices(bounds, num_frames, prefix_states.device, cache=cache)
    before = prefix_states[indices["before_idx"]]
    after = suffix_states[indices["after_idx"]]

    if text_state is not None:
        text = text_state.to(device=before.device, dtype=before.dtype)
        if text.shape != (heads, value_dim, key_dim):
            raise ValueError("text_state has incompatible [heads, value_dim, key_dim] shape")
        shape = (num_frames, 1, 1, 1)
        before = torch.where(indices["has_before"].view(shape), before, text)
        after = torch.where(indices["has_after"].view(shape), after, text)

    if bridge == "alpha":
        log_alpha = alpha.clamp_min(1e-12).log()
        log_prefix = torch.cat((torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)))
        decay_before = torch.exp(
            log_prefix[indices["frames"] + 1]
            - log_prefix[indices["bridge_before"]]
        )
        decay_after = torch.exp(
            log_prefix[indices["bridge_after"]]
            - log_prefix[indices["frames"]]
        )
        # Alpha indexes key channels, hence broadcast over the value axis.
        before = before * decay_before.unsqueeze(2)
        after = after * decay_after.unsqueeze(2)

    if text_state is None:
        mask_shape = (num_frames, 1, 1, 1)
        result = (
            before * indices["has_before"].view(mask_shape)
            + after * indices["has_after"].view(mask_shape)
        )
    else:
        result = before + after
    return result if out_dtype is None else result.to(out_dtype)


def _activate(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    activated = F.silu(tokens)
    if l2norm:
        return F.normalize(activated, dim=-1, eps=1e-6).to(activated.dtype)
    return activated


def _temporal_shift(
    frames: torch.Tensor, weight: torch.Tensor, kernel: int
) -> torch.Tensor:
    """Apply a zero-padded depthwise temporal convolution without Conv1d layout churn."""
    if kernel <= 0 or kernel % 2 == 0 or weight.shape[-1] != kernel:
        raise ValueError("temporal kernel must be positive, odd and match its weight")
    padded = F.pad(frames, (0, 0, 0, 0, kernel // 2, kernel // 2))
    result = torch.zeros_like(frames)
    for offset in range(kernel):
        result = result + padded[offset:offset + frames.shape[0]] * weight[:, offset].view(
            1, 1, -1
        )
    return result


def conv_features(
    tokens: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_weight: torch.Tensor,
    num_frames: int,
    frame_size: tuple[int, int] | None,
    l2norm: bool,
) -> torch.Tensor:
    """Apply spatial then temporal depthwise short convolution and activation."""
    if tokens.ndim != 3 or frame_size is None:
        raise ValueError("conv features need [rows, heads, dim] and frame_size")
    heads, head_dim = tokens.shape[-2:]
    grid_height, grid_width = frame_size
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("frame_size dimensions must be positive")
    channels = heads * head_dim
    if tokens.shape[0] != num_frames * grid_height * grid_width:
        raise ValueError("token rows do not match num_frames * frame area")
    if spatial_weight.shape[0] != channels or spatial_weight.shape[1] != 1:
        raise ValueError("spatial convolution must have one depthwise filter per channel")
    kernel_h, kernel_w = spatial_weight.shape[-2:]
    if kernel_h % 2 == 0 or kernel_w % 2 == 0:
        raise ValueError("spatial kernels must have odd dimensions")

    volume = tokens.reshape(
        num_frames, grid_height, grid_width, channels
    ).permute(0, 3, 1, 2)
    volume = F.conv2d(
        volume, spatial_weight,
        padding=(kernel_h // 2, kernel_w // 2), groups=channels,
    )
    flattened = volume.permute(0, 2, 3, 1).reshape(
        num_frames, grid_height * grid_width, channels
    )
    temporal = temporal_weight.squeeze(1).to(flattened.dtype)
    convolved = _temporal_shift(flattened, temporal, temporal.shape[-1])
    return _activate(convolved.reshape(-1, heads, head_dim), l2norm)


def alpha_gate(
    frame_mean: torch.Tensor,
    weight_down: torch.Tensor,
    weight_up: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute the fp32 KDA double-exponential retention gate."""
    with torch.autocast(device_type=frame_mean.device.type, enabled=False):
        delta = F.linear(frame_mean.float(), weight_down.float())
        delta = F.linear(delta, weight_up.float()) + dt_bias.float()
        if delta.shape[-1] != num_heads * head_dim or a_log.numel() != num_heads:
            raise ValueError("alpha weights are incompatible with head geometry")
        decay_scale = torch.exp(a_log.float()).reshape(num_heads, 1)
        delta = delta.reshape(-1, num_heads, head_dim)
        return torch.exp(-decay_scale * F.softplus(delta))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted RMSNorm with an fp32 norm accumulation."""
    mean_square = torch.linalg.vector_norm(
        x, dim=-1, keepdim=True, dtype=torch.float32
    ).square() / x.shape[-1]
    return x * torch.rsqrt(mean_square + eps).to(x.dtype) * weight.to(x.dtype)


def _linear_epilogue_body(
    readout_fhsd: torch.Tensor,
    norm_weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normalized = rms_norm(readout_fhsd, norm_weight, eps)
    frames, heads, tokens, head_dim = normalized.shape
    rows = frames * tokens
    return normalized.permute(0, 2, 1, 3).reshape(rows, heads * head_dim) * gate.reshape(
        rows, heads * head_dim
    )


class EpilogueCompilerCache:
    """Branch-owned compiled epilogues, releasable with their model."""

    def __init__(self, limit: int = 4):
        if limit <= 0:
            raise ValueError("cache limit must be positive")
        self.limit = int(limit)
        self._compiled: OrderedDict[tuple[str, int | None], Any] = OrderedDict()
        self._broken: set[tuple[str, int | None]] = set()

    def run(self, readout, norm_weight, gate, eps):
        key = _device_key(readout.device)
        if key in self._broken:
            return None
        try:
            compiled = self._compiled.get(key)
            if compiled is None:
                compiled = torch.compile(_linear_epilogue_body)
                self._compiled[key] = compiled
                while len(self._compiled) > self.limit:
                    self._compiled.popitem(last=False)
            else:
                self._compiled.move_to_end(key)
            return compiled(readout, norm_weight, gate, eps)
        except Exception:
            self._broken.add(key)
            self._compiled.pop(key, None)
            return None

    def clear(self) -> None:
        self._compiled.clear()
        self._broken.clear()

    def release(self) -> None:
        self.clear()


def linear_epilogue(
    readout_fhsd: torch.Tensor,
    norm_weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
    fuse: bool = False,
    compiler_cache: EpilogueCompilerCache | None = None,
) -> torch.Tensor:
    """RMSNorm/output gate with opt-in, caller-owned compilation storage."""
    if fuse and compiler_cache is not None:
        compiled_result = compiler_cache.run(readout_fhsd, norm_weight, gate, eps)
        if compiled_result is not None:
            return compiled_result
    return _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps)


class LinearBranch:
    """Checkpoint-backed linear attention branch for one transformer block."""

    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        num_heads: int,
        head_dim: int,
        delta_rule: str = "vdn_solve",
        bridge: str = "alpha",
        a_fp32: bool = True,
        short_conv: Sequence[str] = ("k", "v"),
        enable_text_state: bool = True,
    ):
        if delta_rule not in DELTA_BACKENDS:
            raise ValueError(f"unknown delta rule: {delta_rule!r}")
        if bridge not in ("alpha", "none"):
            raise ValueError("bridge must be 'alpha' or 'none'")
        targets = tuple(short_conv)
        if len(set(targets)) != len(targets) or any(target not in ("q", "k", "v") for target in targets):
            raise ValueError("short_conv must be a distinct subset of ('q', 'k', 'v')")
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("head geometry must be positive")
        self.w = weights
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.delta_rule = delta_rule
        self.bridge = bridge
        self.a_fp32 = a_fp32
        self.short_conv = targets
        self.enable_text_state = enable_text_state
        self.fuse_epilogue = False
        self._backends: OrderedDict[tuple[str, int], VdnDelta | SanaDelta] = OrderedDict()
        self._gather_cache = GatherIndexCache()
        self._epilogue_cache = EpilogueCompilerCache()

    def clear(self) -> None:
        """Release every cached object owned by this branch instance."""
        self._backends.clear()
        self._gather_cache.clear()
        self._epilogue_cache.clear()

    def release(self) -> None:
        self.clear()

    def _delta_backend(self, length: int) -> VdnDelta | SanaDelta:
        key = (self.delta_rule, length)
        backend = self._backends.get(key)
        if backend is None:
            backend = DELTA_BACKENDS[self.delta_rule](length)
            self._backends[key] = backend
            while len(self._backends) > 4:
                self._backends.popitem(last=False)
        else:
            self._backends.move_to_end(key)
        return backend

    def _feature_one(
        self,
        weights: Mapping[str, torch.Tensor],
        tokens: torch.Tensor,
        projection: str,
        num_frames: int,
        frame_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        normalize = projection != "v"
        if projection not in self.short_conv:
            return _activate(tokens, normalize)
        if frame_size is None:
            raise ValueError("short convolution requires frame_size")
        return conv_features(
            tokens,
            weights[f"short_conv.{projection}_sp.weight"],
            weights[f"short_conv.{projection}_tm.weight"],
            num_frames,
            frame_size,
            normalize,
        )

    def _features(
        self,
        weights: Mapping[str, torch.Tensor],
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        num_frames: int,
        frame_size: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            self._feature_one(weights, tensor, projection, num_frames, frame_size)
            for projection, tensor in zip(("q", "k", "v"), (q_raw, k_raw, v_raw))
        )

    def _text_state(
        self,
        weights: Mapping[str, torch.Tensor],
        text_x: torch.Tensor | None,
        text_k_raw: torch.Tensor | None,
        text_v_raw: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.enable_text_state or text_x is None:
            return None
        if text_k_raw is None or text_v_raw is None:
            raise ValueError("text state needs both raw text key and value")
        length = text_x.shape[0]
        if length <= 0 or text_k_raw.shape[0] != length or text_v_raw.shape[0] != length:
            raise ValueError("text tensors must share a positive sequence length")
        key = _activate(text_k_raw, True).reshape(
            1, length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = _activate(text_v_raw, False).reshape(
            1, length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(text_x, weights["beta_proj.weight"]))
        beta = beta.reshape(1, length, self.num_heads).permute(0, 2, 1)
        matrix_a, matrix_b = frame_statistics(key, value, beta, self.a_fp32)
        ones = torch.ones(
            1, self.num_heads, self.head_dim,
            device=matrix_a.device, dtype=matrix_a.dtype,
        )
        _, injection = self._delta_backend(length).factor_apply(ones, matrix_a, matrix_b)
        return TEXT_STATE_SCALE * injection[0]

    def readout(
        self,
        weights: Mapping[str, torch.Tensor],
        xv: torch.Tensor,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
        bounds: Sequence[tuple[int, int]],
        frame_size: tuple[int, int] | None = None,
        text_x: torch.Tensor | None = None,
        text_k_raw: torch.Tensor | None = None,
        text_v_raw: torch.Tensor | None = None,
        skip_ends: bool = False,
    ) -> torch.Tensor:
        """Return gated linear readout for all video rows."""
        expected_rows = num_frames * tokens_per_frame
        if xv.ndim != 2 or xv.shape[0] != expected_rows:
            raise ValueError("xv rows must equal num_frames * tokens_per_frame")
        expected_qkv = (expected_rows, self.num_heads, self.head_dim)
        if q_raw.shape != expected_qkv or k_raw.shape != expected_qkv or v_raw.shape != expected_qkv:
            raise ValueError("raw q/k/v have incompatible video row/head geometry")
        if len(bounds) != num_frames:
            raise ValueError("bounds must contain one entry per frame")

        if skip_ends:
            if num_frames <= 2:
                return xv.new_zeros(expected_rows, self.num_heads * self.head_dim)
            inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
            inner_readout = self._readout(
                weights, xv[inner], (q_raw[inner], k_raw[inner], v_raw[inner]),
                num_frames - 2, tokens_per_frame,
                [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]],
                frame_size, text_x, text_k_raw, text_v_raw,
            )
            output = inner_readout.new_zeros(
                expected_rows, inner_readout.shape[-1]
            )
            output[inner] = inner_readout
            return output
        return self._readout(
            weights, xv, (q_raw, k_raw, v_raw), num_frames,
            tokens_per_frame, bounds, frame_size,
            text_x, text_k_raw, text_v_raw,
        )

    def _readout(
        self,
        weights: Mapping[str, torch.Tensor],
        xv: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        num_frames: int,
        tokens_per_frame: int,
        bounds: Sequence[tuple[int, int]],
        frame_size: tuple[int, int] | None,
        text_x: torch.Tensor | None,
        text_k_raw: torch.Tensor | None,
        text_v_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        rows = num_frames * tokens_per_frame
        shape = (num_frames, tokens_per_frame, self.num_heads, self.head_dim)
        query, key, value = self._features(
            weights, *qkv_raw, num_frames, frame_size
        )
        query_by_frame = query.reshape(shape).permute(0, 2, 1, 3)
        key_by_frame = key.reshape(shape).permute(0, 2, 1, 3)
        value_by_frame = value.reshape(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(xv, weights["beta_proj.weight"]))
        beta = beta.reshape(num_frames, tokens_per_frame, self.num_heads).permute(0, 2, 1)
        matrix_a, matrix_b = frame_statistics(
            key_by_frame, value_by_frame, beta, self.a_fp32
        )
        frame_mean = xv.reshape(num_frames, tokens_per_frame, -1).mean(
            dim=1, dtype=torch.float32
        )
        alpha = alpha_gate(
            frame_mean,
            weights["alpha.down.weight"],
            weights["alpha.up.weight"],
            weights["alpha.dt_bias"],
            weights["alpha.A_log"],
            self.num_heads,
            self.head_dim,
        )
        text_state = self._text_state(
            weights, text_x, text_k_raw, text_v_raw
        )
        prefix, suffix = run_scans(
            self._delta_backend(tokens_per_frame), alpha, matrix_a, matrix_b,
            text_state=text_state,
        )
        hidden_gate = F.linear(xv, weights["output_gate.down.weight"])
        gate = torch.sigmoid(
            F.linear(
                hidden_gate,
                weights["output_gate.up.weight"],
                weights.get("output_gate.up.bias"),
            )
        )
        linear_state = gather_linear_state(
            prefix, suffix, alpha, bounds, bridge=self.bridge,
            text_state=text_state, out_dtype=gate.dtype, cache=self._gather_cache,
        )
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        return linear_epilogue(
            readout, weights["norm.weight"], gate, 1e-6,
            fuse=self.fuse_epilogue, compiler_cache=self._epilogue_cache,
        ).reshape(rows, self.num_heads * self.head_dim)


__all__ = [
    "DELTA_BACKENDS",
    "EpilogueCompilerCache",
    "GatherIndexCache",
    "LinearBranch",
    "SanaDelta",
    "TEXT_STATE_SCALE",
    "VdnDelta",
    "alpha_gate",
    "conv_features",
    "frame_statistics",
    "gather_indices",
    "gather_linear_state",
    "linear_epilogue",
    "rms_norm",
    "run_scans",
]
