"""VDN bidirectional linear branch with separate reference and inference bodies.

The reference spelling remains autograd-safe and numerically conservative. The
inference spelling ports the memory-saving scan, fused preparation/gather/epilogue
strategy and temporal-kernel hierarchy used by OpenVDN while keeping every cache
owned by one branch/model.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from .kernels import LinearKernelCache, activate


TEXT_STATE_SCALE = 0.5


class GatherIndexCache:
    def __init__(self, limit: int = 32):
        if limit <= 0:
            raise ValueError("cache limit must be positive")
        self.limit = int(limit)
        self._entries: OrderedDict[tuple[Any, ...], dict[str, torch.Tensor]] = OrderedDict()

    def get(self, key):
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key, value):
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)

    def clear(self):
        self._entries.clear()

    release = clear

    def __len__(self):
        return len(self._entries)


class BranchCompilerCache:
    """Model-owned static-shape torch.compile cache with failure latching."""

    def __init__(self, limit: int = 16):
        self.limit = int(limit)
        self._compiled: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._broken: set[tuple[Any, ...]] = set()

    def call(self, name: str, fn, args: tuple, key: tuple[Any, ...]):
        full_key = (name, *key)
        if full_key in self._broken:
            return None
        compiled = self._compiled.get(full_key)
        if compiled is None:
            try:
                compiled = torch.compile(fn, dynamic=False)
            except Exception:
                self._broken.add(full_key)
                return None
            self._compiled[full_key] = compiled
            while len(self._compiled) > self.limit:
                self._compiled.popitem(last=False)
        else:
            self._compiled.move_to_end(full_key)
        try:
            return compiled(*args)
        except Exception:
            self._broken.add(full_key)
            self._compiled.pop(full_key, None)
            return None

    def clear(self):
        self._compiled.clear()
        self._broken.clear()

    release = clear


class EpilogueCompilerCache(BranchCompilerCache):
    """Compatibility name retained for existing tests/importers."""


class VdnDelta:
    def __init__(self, tokens_per_frame: int | None = None):
        del tokens_per_frame

    def factor_apply(self, alpha, a_raw, b_raw):
        if a_raw.shape != b_raw.shape or a_raw.shape[-1] != a_raw.shape[-2]:
            raise ValueError("A and B must share a square [..., D, D] shape")
        if alpha.shape != a_raw.shape[:-1]:
            raise ValueError("alpha must have shape A.shape[:-1]")
        a32, b32 = a_raw.float(), b_raw.float()
        identity = torch.eye(a32.shape[-1], device=a32.device, dtype=torch.float32).expand_as(a32)
        chol = torch.linalg.cholesky(a32 + identity)
        inv_chol = torch.linalg.solve_triangular(chol, identity, upper=False, left=True)
        inverse = inv_chol.transpose(-1, -2) @ inv_chol
        transition = alpha.float().unsqueeze(-1) * inverse
        injection = b32 @ inverse
        return transition, injection


class SanaDelta:
    def __init__(self, tokens_per_frame: int):
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        self.inv_tokens = 1.0 / tokens_per_frame
        self.inv_sqrt_tokens = self.inv_tokens**0.5

    def factor_apply(self, alpha, a_raw, b_raw):
        if a_raw.shape != b_raw.shape or a_raw.shape[-1] != a_raw.shape[-2]:
            raise ValueError("A and B must share a square [..., D, D] shape")
        if alpha.shape != a_raw.shape[:-1]:
            raise ValueError("alpha must have shape A.shape[:-1]")
        identity = torch.eye(a_raw.shape[-1], device=a_raw.device, dtype=a_raw.dtype)
        transition = alpha.unsqueeze(-1) * (identity - self.inv_tokens * a_raw)
        return transition, self.inv_sqrt_tokens * b_raw


DELTA_BACKENDS = {"vdn_solve": VdnDelta, "sana_scaled": SanaDelta}


@contextmanager
def _tf32_matmul(device: torch.device | None = None):
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


def _frame_stats_prep_body(key_by_frame, value_by_frame, beta):
    key16 = key_by_frame.contiguous()
    key32 = key16.float()
    weighted_key32 = (key32 * beta.float().unsqueeze(-1)).contiguous()
    weighted_value = (value_by_frame * beta.to(value_by_frame.dtype).unsqueeze(-1)).contiguous()
    return key16, key32, weighted_key32, weighted_value


def frame_statistics(
    key_by_frame: torch.Tensor,
    value_by_frame: torch.Tensor,
    beta: torch.Tensor,
    a_fp32: bool = True,
    *,
    inference: bool = False,
    compiler_cache: BranchCompilerCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if key_by_frame.ndim != 4 or value_by_frame.shape != key_by_frame.shape:
        raise ValueError("key and value must share shape [frames, heads, tokens, dim]")
    if beta.shape != key_by_frame.shape[:-1]:
        raise ValueError("beta must have shape [frames, heads, tokens]")
    with torch.autocast(device_type=key_by_frame.device.type, enabled=False):
        prepared = None
        if inference and key_by_frame.is_cuda and compiler_cache is not None:
            key = (
                key_by_frame.device.type, key_by_frame.device.index, key_by_frame.dtype,
                tuple(key_by_frame.shape), value_by_frame.dtype, beta.dtype,
            )
            prepared = compiler_cache.call(
                "frame_stats_prep", _frame_stats_prep_body,
                (key_by_frame, value_by_frame, beta), key,
            )
        if prepared is None:
            prepared = _frame_stats_prep_body(key_by_frame, value_by_frame, beta)
        key16, key32, weighted_key32, weighted_value = prepared
        if a_fp32:
            with _tf32_matmul(key_by_frame.device) if inference else nullcontext():
                matrix_a = weighted_key32.transpose(-1, -2) @ key32
        else:
            weighted_key = (key16 * beta.to(key16.dtype).unsqueeze(-1)).contiguous()
            matrix_a = (weighted_key.transpose(-1, -2) @ key16).float()
        matrix_a = 0.5 * (matrix_a + matrix_a.transpose(-1, -2))
        matrix_b = (weighted_value.transpose(-1, -2) @ key16).float()
    return matrix_a, matrix_b


def _scan_inputs(backend, alpha, a_raw, b_raw, text_state):
    transitions, injections = backend.factor_apply(alpha, a_raw, b_raw)
    initial = (
        torch.zeros_like(injections[0])
        if text_state is None
        else text_state.to(device=injections.device, dtype=injections.dtype)
    )
    if initial.shape != injections.shape[1:]:
        raise ValueError("text_state must have shape [heads, D, D]")
    return transitions, injections, initial


def run_scans_reference(backend, alpha, a_raw, b_raw, text_state=None):
    if a_raw.ndim != 4 or a_raw.shape != b_raw.shape or a_raw.shape[0] == 0:
        raise ValueError("A and B must share non-empty [frames, heads, D, D] shape")
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections, initial = _scan_inputs(backend, alpha, a_raw, b_raw, text_state)
        forward_states = []
        state = initial
        for frame in range(transitions.shape[0]):
            state = torch.baddbmm(injections[frame], state, transitions[frame])
            forward_states.append(state)
        reverse_states = [initial] * transitions.shape[0]
        state = initial
        for frame in range(transitions.shape[0] - 1, -1, -1):
            state = torch.baddbmm(injections[frame], state, transitions[frame])
            reverse_states[frame] = state
        return torch.stack(forward_states), torch.stack(reverse_states)


def run_scans_inference(backend, alpha, a_raw, b_raw, text_state=None):
    if torch.is_grad_enabled():
        raise RuntimeError("VDN inference scan requires torch.no_grad()/inference_mode()")
    if a_raw.ndim != 4 or a_raw.shape != b_raw.shape or a_raw.shape[0] == 0:
        raise ValueError("A and B must share non-empty [frames, heads, D, D] shape")
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections, initial = _scan_inputs(backend, alpha, a_raw, b_raw, text_state)
        frames = transitions.shape[0]
        prefix = torch.empty((frames, *initial.shape), dtype=injections.dtype, device=injections.device)
        suffix = torch.empty_like(prefix)
        state = initial
        for frame in range(frames):
            torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
            state = prefix[frame]
        state = initial
        for frame in range(frames - 1, -1, -1):
            torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
            state = suffix[frame]
        return prefix, suffix


def run_scans(backend, alpha, a_raw, b_raw, text_state=None):
    return run_scans_reference(backend, alpha, a_raw, b_raw, text_state=text_state)


def _device_key(device: torch.device | str):
    resolved = torch.device(device)
    return resolved.type, resolved.index


def gather_indices(bounds, num_frames, device, cache: GatherIndexCache | None = None):
    if num_frames <= 0 or len(bounds) != num_frames:
        raise ValueError("bounds must contain one entry for every positive frame count")
    normalized = tuple((int(lo), int(hi)) for lo, hi in bounds)
    for frame, (lo, hi) in enumerate(normalized):
        if lo > frame or hi < frame or lo > hi:
            raise ValueError(f"bounds[{frame}] does not contain its query frame")
    cache_key = (normalized, num_frames, _device_key(device))
    cached = cache.get(cache_key) if cache is not None else None
    if cached is not None:
        return cached
    last_before = torch.tensor([lo for lo, _ in normalized], dtype=torch.long, device=device) - 1
    first_after = torch.tensor([hi for _, hi in normalized], dtype=torch.long, device=device) + 1
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
        cache.put(cache_key, cached)
    return cached


def _gather_body(
    prefix_states, suffix_states, alpha, text_state, use_alpha, out_dtype,
    before_idx, after_idx, has_before, has_after, bridge_before, bridge_after, frames,
):
    num_frames = prefix_states.shape[0]
    before = prefix_states[before_idx]
    after = suffix_states[after_idx]
    if text_state is not None:
        text = text_state.to(device=before.device, dtype=before.dtype)
        shape = (num_frames, 1, 1, 1)
        before = torch.where(has_before.view(shape), before, text)
        after = torch.where(has_after.view(shape), after, text)
    if use_alpha:
        log_alpha = alpha.clamp_min(1e-12).log()
        log_prefix = torch.cat((torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)))
        decay_before = torch.exp(log_prefix[frames + 1] - log_prefix[bridge_before])
        decay_after = torch.exp(log_prefix[bridge_after] - log_prefix[frames])
        before = before * decay_before.unsqueeze(2)
        after = after * decay_after.unsqueeze(2)
    if text_state is None:
        shape = (num_frames, 1, 1, 1)
        result = before * has_before.view(shape) + after * has_after.view(shape)
    else:
        result = before + after
    return result if out_dtype is None else result.to(out_dtype)


def gather_linear_state(
    prefix_states,
    suffix_states,
    alpha,
    bounds,
    bridge="alpha",
    text_state=None,
    out_dtype=None,
    cache: GatherIndexCache | None = None,
    *,
    inference: bool = False,
    compiler_cache: BranchCompilerCache | None = None,
):
    if bridge not in ("alpha", "none"):
        raise ValueError("bridge must be 'alpha' or 'none'")
    if prefix_states.ndim != 4 or prefix_states.shape != suffix_states.shape:
        raise ValueError("prefix and suffix must share [frames, heads, D, D] shape")
    frames, heads, value_dim, key_dim = prefix_states.shape
    if alpha.shape != (frames, heads, key_dim):
        raise ValueError("alpha has incompatible [frames, heads, key_dim] shape")
    if text_state is not None and text_state.shape != (heads, value_dim, key_dim):
        raise ValueError("text_state has incompatible [heads, value_dim, key_dim] shape")
    idx = gather_indices(bounds, frames, prefix_states.device, cache=cache)
    args = (
        prefix_states, suffix_states, alpha, text_state, bridge == "alpha", out_dtype,
        idx["before_idx"], idx["after_idx"], idx["has_before"], idx["has_after"],
        idx["bridge_before"], idx["bridge_after"], idx["frames"],
    )
    if inference and prefix_states.is_cuda and compiler_cache is not None:
        key = (
            _device_key(prefix_states.device), prefix_states.dtype, tuple(prefix_states.shape),
            alpha.dtype, bridge, text_state is not None, out_dtype,
        )
        compiled = compiler_cache.call("gather", _gather_body, args, key)
        if compiled is not None:
            return compiled
    return _gather_body(*args)


def _temporal_shift(frames, weight, kernel):
    from .kernels import temporal_shift_eager
    return temporal_shift_eager(frames, weight, kernel, kernel // 2)


def conv_features(
    tokens,
    spatial_weight,
    temporal_weight,
    num_frames,
    frame_size,
    l2norm,
    *,
    kernel_mode="eager",
    kernel_cache: LinearKernelCache | None = None,
    fhsd=False,
):
    if tokens.ndim != 3 or frame_size is None:
        raise ValueError("conv features need [rows, heads, dim] and frame_size")
    heads, head_dim = tokens.shape[-2:]
    gh, gw = frame_size
    if gh <= 0 or gw <= 0:
        raise ValueError("frame_size dimensions must be positive")
    channels = heads * head_dim
    if tokens.shape[0] != num_frames * gh * gw:
        raise ValueError("token rows do not match num_frames * frame area")
    if spatial_weight.shape[0] != channels or spatial_weight.shape[1] != 1:
        raise ValueError("spatial convolution must have one depthwise filter per channel")
    kh, kw = spatial_weight.shape[-2:]
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("spatial kernels must have odd dimensions")
    volume = tokens.reshape(num_frames, gh, gw, channels).permute(0, 3, 1, 2)
    volume = F.conv2d(volume, spatial_weight, padding=(kh // 2, kw // 2), groups=channels)
    flattened = volume.permute(0, 2, 3, 1).reshape(num_frames, gh * gw, channels)
    temporal = temporal_weight.squeeze(1).to(flattened.dtype).contiguous()
    cache = kernel_cache or LinearKernelCache(limit=1)
    result = cache.temporal(
        flattened.contiguous(), temporal, temporal.shape[-1], temporal.shape[-1] // 2,
        heads, head_dim, l2norm, mode=kernel_mode,
    )
    if fhsd:
        return result.reshape(num_frames, gh * gw, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return result


def alpha_gate(frame_mean, weight_down, weight_up, dt_bias, a_log, num_heads, head_dim):
    with torch.autocast(device_type=frame_mean.device.type, enabled=False):
        delta = F.linear(frame_mean.float(), weight_down.float())
        delta = F.linear(delta, weight_up.float()) + dt_bias.float()
        if delta.shape[-1] != num_heads * head_dim or a_log.numel() != num_heads:
            raise ValueError("alpha weights are incompatible with head geometry")
        decay_scale = torch.exp(a_log.float()).reshape(num_heads, 1)
        delta = delta.reshape(-1, num_heads, head_dim)
        return torch.exp(-decay_scale * F.softplus(delta))


def rms_norm(x, weight, eps=1e-6):
    mean_square = torch.linalg.vector_norm(
        x, dim=-1, keepdim=True, dtype=torch.float32
    ).square() / x.shape[-1]
    return x * torch.rsqrt(mean_square + eps).to(x.dtype) * weight.to(x.dtype)


def _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps):
    normalized = rms_norm(readout_fhsd, norm_weight, eps)
    frames, heads, tokens, head_dim = normalized.shape
    rows = frames * tokens
    return normalized.permute(0, 2, 1, 3).reshape(rows, heads * head_dim) * gate.reshape(rows, heads * head_dim)


def linear_epilogue(
    readout_fhsd,
    norm_weight,
    gate,
    eps=1e-6,
    fuse=False,
    compiler_cache: BranchCompilerCache | None = None,
    *,
    inference: bool | None = None,
):
    use_compile = bool(fuse if inference is None else inference)
    if use_compile and readout_fhsd.is_cuda and compiler_cache is not None:
        key = (
            _device_key(readout_fhsd.device), readout_fhsd.dtype,
            tuple(readout_fhsd.shape), norm_weight.dtype, gate.dtype, tuple(gate.shape), eps,
        )
        result = compiler_cache.call(
            "epilogue", _linear_epilogue_body,
            (readout_fhsd, norm_weight, gate, eps), key,
        )
        if result is not None:
            return result
    return _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps)


class LinearBranch:
    def __init__(
        self,
        weights: Mapping[str, torch.Tensor] | None,
        num_heads: int,
        head_dim: int,
        delta_rule="vdn_solve",
        bridge="alpha",
        a_fp32=True,
        short_conv=("k", "v"),
        enable_text_state=True,
        *,
        linear_kernels="auto",
    ):
        if delta_rule not in DELTA_BACKENDS:
            raise ValueError(f"unknown delta rule: {delta_rule!r}")
        if bridge not in ("alpha", "none"):
            raise ValueError("bridge must be 'alpha' or 'none'")
        targets = tuple(short_conv)
        if len(set(targets)) != len(targets) or any(t not in ("q", "k", "v") for t in targets):
            raise ValueError("short_conv must be a distinct subset of ('q', 'k', 'v')")
        if linear_kernels not in {"auto", "triton", "compile", "conv1d", "eager"}:
            raise ValueError(f"unknown linear_kernels {linear_kernels!r}")
        self.w = weights
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.delta_rule = delta_rule
        self.bridge = bridge
        self.a_fp32 = bool(a_fp32)
        self.short_conv = targets
        self.enable_text_state = bool(enable_text_state)
        self.linear_kernels = linear_kernels
        self.fuse_epilogue = linear_kernels != "eager"
        self._backends: OrderedDict[tuple[str, int], VdnDelta | SanaDelta] = OrderedDict()
        self._gather_cache = GatherIndexCache()
        self._compiler_cache = BranchCompilerCache()
        self._epilogue_cache = self._compiler_cache
        self._kernel_cache = LinearKernelCache()
        self.diagnostics = None

    def detach_weights(self):
        self.w = None

    def set_runtime(self, *, linear_kernels: str | None = None, diagnostics=None):
        if linear_kernels is not None:
            self.linear_kernels = linear_kernels
            self.fuse_epilogue = linear_kernels != "eager"
        self.diagnostics = diagnostics
        return self

    def _scope(self, name: str, device):
        if self.diagnostics is None:
            return nullcontext()
        return self.diagnostics.scope(f"linear.{name}", device)

    def clear(self):
        self._backends.clear()
        self._gather_cache.clear()
        self._compiler_cache.clear()
        self._kernel_cache.release()

    release = clear

    def _delta_backend(self, length):
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

    def _feature_one(self, weights, tokens, projection, num_frames, frame_size, *, inference=False, fhsd=False):
        normalize = projection != "v"
        if projection in self.short_conv:
            if frame_size is None:
                raise ValueError("short convolution requires frame_size")
            return conv_features(
                tokens,
                weights[f"short_conv.{projection}_sp.weight"],
                weights[f"short_conv.{projection}_tm.weight"],
                num_frames,
                frame_size,
                normalize,
                kernel_mode=(self.linear_kernels if inference else "eager"),
                kernel_cache=self._kernel_cache,
                fhsd=fhsd,
            )
        if inference:
            return self._kernel_cache.activate(
                tokens, normalize,
                num_frames=num_frames,
                per_frame=(tokens.shape[0] // num_frames if num_frames else None),
                fhsd=fhsd,
                compile_enabled=self.linear_kernels != "eager",
            )
        result = activate(tokens, normalize)
        if fhsd:
            per_frame = tokens.shape[0] // num_frames
            return result.reshape(num_frames, per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return result

    def _features(self, weights, q_raw, k_raw, v_raw, num_frames, frame_size, *, inference=False):
        if inference:
            q = self._feature_one(weights, q_raw, "q", num_frames, frame_size, inference=True, fhsd=True)
            k = self._feature_one(weights, k_raw, "k", num_frames, frame_size, inference=True)
            v = self._feature_one(weights, v_raw, "v", num_frames, frame_size, inference=True)
            return q, k, v
        return tuple(
            self._feature_one(weights, tensor, projection, num_frames, frame_size)
            for projection, tensor in zip(("q", "k", "v"), (q_raw, k_raw, v_raw))
        )

    def _text_state(self, weights, text_x, text_k_raw, text_v_raw):
        if not self.enable_text_state or text_x is None:
            return None
        if text_k_raw is None or text_v_raw is None:
            raise ValueError("text state needs both raw text key and value")
        length = text_x.shape[0]
        if length <= 0 or text_k_raw.shape[0] != length or text_v_raw.shape[0] != length:
            raise ValueError("text tensors must share a positive sequence length")
        key = activate(text_k_raw, True).reshape(
            1, length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = activate(text_v_raw, False).reshape(
            1, length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(text_x, weights["beta_proj.weight"]))
        beta = beta.reshape(1, length, self.num_heads).permute(0, 2, 1)
        matrix_a, matrix_b = frame_statistics(key, value, beta, self.a_fp32)
        ones = torch.ones(
            1, self.num_heads, self.head_dim, device=matrix_a.device, dtype=matrix_a.dtype
        )
        _, injection = self._delta_backend(length).factor_apply(ones, matrix_a, matrix_b)
        return TEXT_STATE_SCALE * injection[0]

    def readout(
        self,
        weights,
        xv,
        q_raw,
        k_raw,
        v_raw,
        num_frames,
        tokens_per_frame,
        bounds,
        frame_size=None,
        text_x=None,
        text_k_raw=None,
        text_v_raw=None,
        skip_ends=False,
        *,
        inference=False,
    ):
        if inference and torch.is_grad_enabled():
            raise RuntimeError("VDN-H3 inference linear branch requires no_grad/inference_mode")
        expected_rows = num_frames * tokens_per_frame
        if xv.ndim != 2 or xv.shape[0] != expected_rows:
            raise ValueError("xv rows must equal num_frames * tokens_per_frame")
        expected_qkv = (expected_rows, self.num_heads, self.head_dim)
        if q_raw.shape != expected_qkv or k_raw.shape != expected_qkv or v_raw.shape != expected_qkv:
            raise ValueError("raw q/k/v have incompatible video row/head geometry")
        if len(bounds) != num_frames:
            raise ValueError("bounds must contain one entry per frame")
        run = self._readout_inference if inference else self._readout_reference
        if not skip_ends:
            return run(
                weights, xv, (q_raw, k_raw, v_raw), num_frames, tokens_per_frame,
                bounds, frame_size, text_x, text_k_raw, text_v_raw,
            )
        if num_frames <= 2:
            return xv.new_zeros(expected_rows, self.num_heads * self.head_dim)
        inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
        inner_readout = run(
            weights, xv[inner], (q_raw[inner], k_raw[inner], v_raw[inner]),
            num_frames - 2, tokens_per_frame,
            [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]],
            frame_size, text_x, text_k_raw, text_v_raw,
        )
        if inference:
            output = inner_readout.new_empty(expected_rows, inner_readout.shape[-1])
            output[:tokens_per_frame].zero_()
            output[(num_frames - 1) * tokens_per_frame:].zero_()
        else:
            output = inner_readout.new_zeros(expected_rows, inner_readout.shape[-1])
        output[inner] = inner_readout
        return output

    def _common_statistics(self, weights, xv, qkv_raw, num_frames, tokens_per_frame, frame_size, *, inference):
        shape = (num_frames, tokens_per_frame, self.num_heads, self.head_dim)
        with self._scope("features", xv.device):
            query, key, value = self._features(
                weights, *qkv_raw, num_frames, frame_size, inference=inference
            )
        if inference:
            query_by_frame = query
        else:
            query_by_frame = query.reshape(shape).permute(0, 2, 1, 3)
        key_by_frame = key.reshape(shape).permute(0, 2, 1, 3)
        value_by_frame = value.reshape(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(xv, weights["beta_proj.weight"]))
        beta = beta.reshape(num_frames, tokens_per_frame, self.num_heads).permute(0, 2, 1)
        with self._scope("frame_stats", xv.device):
            matrix_a, matrix_b = frame_statistics(
                key_by_frame, value_by_frame, beta, self.a_fp32,
                inference=inference, compiler_cache=self._compiler_cache,
            )
        frame_mean = xv.reshape(num_frames, tokens_per_frame, -1).mean(
            dim=1, dtype=torch.float32
        )
        alpha = alpha_gate(
            frame_mean,
            weights["alpha.down.weight"], weights["alpha.up.weight"],
            weights["alpha.dt_bias"], weights["alpha.A_log"],
            self.num_heads, self.head_dim,
        )
        return query_by_frame, matrix_a, matrix_b, alpha

    def _output_gate(self, weights, xv):
        hidden = F.linear(xv, weights["output_gate.down.weight"])
        return torch.sigmoid(
            F.linear(hidden, weights["output_gate.up.weight"], weights.get("output_gate.up.bias"))
        )

    def _readout_reference(
        self, weights, xv, qkv_raw, num_frames, tokens_per_frame, bounds,
        frame_size, text_x, text_k_raw, text_v_raw,
    ):
        rows = num_frames * tokens_per_frame
        query_by_frame, matrix_a, matrix_b, alpha = self._common_statistics(
            weights, xv, qkv_raw, num_frames, tokens_per_frame, frame_size, inference=False
        )
        text_state = self._text_state(weights, text_x, text_k_raw, text_v_raw)
        with self._scope("scan", xv.device):
            prefix, suffix = run_scans_reference(
                self._delta_backend(tokens_per_frame), alpha, matrix_a, matrix_b,
                text_state=text_state,
            )
        gate = self._output_gate(weights, xv)
        with self._scope("gather", xv.device):
            linear_state = gather_linear_state(
                prefix, suffix, alpha, bounds, bridge=self.bridge, text_state=text_state,
                out_dtype=gate.dtype, cache=self._gather_cache,
            )
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        with self._scope("epilogue", xv.device):
            return linear_epilogue(
                readout, weights["norm.weight"], gate, 1e-6,
                fuse=False, compiler_cache=self._compiler_cache,
            ).reshape(rows, self.num_heads * self.head_dim)

    def _readout_inference(
        self, weights, xv, qkv_raw, num_frames, tokens_per_frame, bounds,
        frame_size, text_x, text_k_raw, text_v_raw,
    ):
        rows = num_frames * tokens_per_frame
        query_by_frame, matrix_a, matrix_b, alpha = self._common_statistics(
            weights, xv, qkv_raw, num_frames, tokens_per_frame, frame_size, inference=True
        )
        text_state = self._text_state(weights, text_x, text_k_raw, text_v_raw)
        with self._scope("scan", xv.device):
            prefix, suffix = run_scans_inference(
                self._delta_backend(tokens_per_frame), alpha, matrix_a, matrix_b,
                text_state=text_state,
            )
        del matrix_a, matrix_b
        gate = self._output_gate(weights, xv)
        with self._scope("gather", xv.device):
            linear_state = gather_linear_state(
                prefix, suffix, alpha, bounds, bridge=self.bridge, text_state=text_state,
                out_dtype=gate.dtype, cache=self._gather_cache,
                inference=True, compiler_cache=self._compiler_cache,
            )
        del prefix, suffix
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        del query_by_frame, linear_state
        with self._scope("epilogue", xv.device):
            return linear_epilogue(
                readout, weights["norm.weight"], gate, 1e-6,
                fuse=True, compiler_cache=self._compiler_cache, inference=True,
            ).reshape(rows, self.num_heads * self.head_dim)


__all__ = [
    "BranchCompilerCache",
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
    "run_scans_inference",
    "run_scans_reference",
]
