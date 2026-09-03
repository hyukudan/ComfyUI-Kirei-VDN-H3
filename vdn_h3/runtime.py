"""Shared and low-activation-memory runtime for the VDN linear branch."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from . import branch as reference
from .kernels import COMPILE_POLICIES, KERNEL_BACKENDS, LinearKernelCache


class SharedCompilerCache:
    """One torch.compile cache shared by all 50 H3 blocks of a patched model."""

    def __init__(self, limit: int = 64, *, policy: str = "shared"):
        if policy not in COMPILE_POLICIES:
            raise ValueError(f"unknown compile policy {policy!r}")
        self.limit = int(limit)
        self.policy = policy
        self._compiled: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._broken: set[tuple[Any, ...]] = set()

    @staticmethod
    def _kwargs(policy: str):
        if policy == "reduce_overhead":
            return {"mode": "reduce-overhead"}
        if policy == "max_autotune":
            return {"mode": "max-autotune"}
        return {}

    def set_policy(self, policy: str):
        if policy not in COMPILE_POLICIES:
            raise ValueError(f"unknown compile policy {policy!r}")
        self.policy = policy

    def call(self, name: str, fn, args: tuple, key: tuple[Any, ...], *, policy: str | None = None):
        policy = self.policy if policy is None else policy
        full_key = (name, policy, *key)
        if policy == "off" or full_key in self._broken:
            return None
        compiled = self._compiled.get(full_key)
        if compiled is None:
            try:
                compiled = torch.compile(fn, dynamic=False, **self._kwargs(policy))
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


class SharedBranchRuntime:
    """Caches and policies shared across every transformed H3 block."""

    def __init__(
        self,
        *,
        kernel_backend: str = "auto",
        compile_policy: str = "shared",
        tile_frames: int = 0,
    ):
        if kernel_backend == "compile":
            kernel_backend = "auto"
            if compile_policy == "off":
                compile_policy = "shared"
        if kernel_backend not in KERNEL_BACKENDS:
            raise ValueError(f"unknown kernel backend {kernel_backend!r}")
        if compile_policy not in COMPILE_POLICIES:
            raise ValueError(f"unknown compile policy {compile_policy!r}")
        if tile_frames < 0:
            raise ValueError("tile_frames must be >= 0")
        self.kernel_backend = kernel_backend
        self.compile_policy = compile_policy
        self.tile_frames = int(tile_frames)
        self.gather = reference.GatherIndexCache(limit=64)
        self.compiler = SharedCompilerCache(limit=96, policy=compile_policy)
        self.kernels = LinearKernelCache(limit=64, compile_policy=compile_policy)
        self.delta_backends: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def delta_backend(self, rule: str, length: int):
        key = (rule, int(length))
        backend = self.delta_backends.get(key)
        if backend is None:
            backend = reference.DELTA_BACKENDS[rule](length)
            self.delta_backends[key] = backend
            while len(self.delta_backends) > 8:
                self.delta_backends.popitem(last=False)
        else:
            self.delta_backends.move_to_end(key)
        return backend

    def release(self):
        self.gather.release()
        self.compiler.release()
        self.kernels.release()
        self.delta_backends.clear()


def _scan_recurrence_body(transitions, injections, initial):
    """Official forward-only recurrence: one baddbmm launch per frame/direction."""
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


def _run_scans_shared(backend, alpha, matrix_a, matrix_b, text_state, cache: SharedBranchRuntime):
    """Factor all frames in batch, then run the launch-light serial recurrence.

    The upstream tuned implementation deliberately does *not* hand the whole Python
    recurrence to Inductor. Static-compiling 2*F dependent baddbmm calls produces a
    large graph, adds first-run/re-specialisation cost and has not shown a steady-state
    advantage over the preallocated one-launch-per-frame loop. Compilation remains on
    the wide pointwise/gather/prologue work where fusion actually removes HBM passes.
    """
    del cache
    if torch.is_grad_enabled():
        raise RuntimeError("VDN inference scan requires torch.no_grad()/inference_mode()")
    with torch.autocast(device_type=matrix_a.device.type, enabled=False):
        transitions, injections, initial = reference._scan_inputs(  # noqa: SLF001
            backend, alpha, matrix_a, matrix_b, text_state
        )
        return _scan_recurrence_body(transitions, injections, initial)


def _gather_range_body(
    prefix_states,
    suffix_states,
    alpha,
    text_state,
    use_alpha,
    out_dtype,
    before_idx,
    after_idx,
    has_before,
    has_after,
    bridge_before,
    bridge_after,
    frame_ids,
):
    count = frame_ids.shape[0]
    before = prefix_states[before_idx]
    after = suffix_states[after_idx]
    shape = (count, 1, 1, 1)
    if text_state is not None:
        text = text_state.to(device=before.device, dtype=before.dtype)
        before = torch.where(has_before.view(shape), before, text)
        after = torch.where(has_after.view(shape), after, text)
    if use_alpha:
        log_alpha = alpha.clamp_min(1e-12).log()
        log_prefix = torch.cat((torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)))
        decay_before = torch.exp(log_prefix[frame_ids + 1] - log_prefix[bridge_before])
        decay_after = torch.exp(log_prefix[bridge_after] - log_prefix[frame_ids])
        before = before * decay_before.unsqueeze(2)
        after = after * decay_after.unsqueeze(2)
    if text_state is None:
        result = before * has_before.view(shape) + after * has_after.view(shape)
    else:
        result = before + after
    return result if out_dtype is None else result.to(out_dtype)


def _gather_range(
    prefix,
    suffix,
    alpha,
    bounds,
    start,
    stop,
    *,
    bridge,
    text_state,
    out_dtype,
    runtime: SharedBranchRuntime,
):
    indices = reference.gather_indices(bounds, prefix.shape[0], prefix.device, cache=runtime.gather)
    sl = slice(start, stop)
    args = (
        prefix,
        suffix,
        alpha,
        text_state,
        bridge == "alpha",
        out_dtype,
        indices["before_idx"][sl],
        indices["after_idx"][sl],
        indices["has_before"][sl],
        indices["has_after"][sl],
        indices["bridge_before"][sl],
        indices["bridge_after"][sl],
        indices["frames"][sl],
    )
    if prefix.is_cuda and runtime.compile_policy != "off":
        key = (
            reference._device_key(prefix.device),  # noqa: SLF001
            prefix.dtype,
            tuple(prefix.shape),
            int(stop - start),
            alpha.dtype,
            bridge,
            text_state is not None,
            out_dtype,
        )
        result = runtime.compiler.call("gather_range", _gather_range_body, args, key)
        if result is not None:
            return result
    return _gather_range_body(*args)


class OptimizedLinearBranch(reference.LinearBranch):
    """LinearBranch with shared caches and an exact low-activation-memory tiled path."""

    def __init__(self, *args, kernel_backend="auto", compile_policy="shared", tile_frames=0, **kwargs):
        super().__init__(*args, linear_kernels=kernel_backend, **kwargs)
        self.kernel_backend = kernel_backend
        self.compile_policy = compile_policy
        self.tile_frames = int(tile_frames)
        self._shared_runtime: SharedBranchRuntime | None = None

    def set_runtime(
        self,
        *,
        runtime_cache: SharedBranchRuntime | None = None,
        kernel_backend: str | None = None,
        compile_policy: str | None = None,
        tile_frames: int | None = None,
        linear_kernels: str | None = None,
        diagnostics=None,
    ):
        if linear_kernels is not None and kernel_backend is None:
            kernel_backend = linear_kernels
        if kernel_backend == "compile":
            kernel_backend = "auto"
            if compile_policy in {None, "off"}:
                compile_policy = "shared"
        if kernel_backend is not None:
            if kernel_backend not in KERNEL_BACKENDS:
                raise ValueError(f"unknown kernel backend {kernel_backend!r}")
            self.kernel_backend = kernel_backend
            self.linear_kernels = kernel_backend
        if compile_policy is not None:
            if compile_policy not in COMPILE_POLICIES:
                raise ValueError(f"unknown compile policy {compile_policy!r}")
            self.compile_policy = compile_policy
        if tile_frames is not None:
            if tile_frames < 0:
                raise ValueError("tile_frames must be >= 0")
            self.tile_frames = int(tile_frames)
        if runtime_cache is not None:
            self._shared_runtime = runtime_cache
            self._gather_cache = runtime_cache.gather
            self._compiler_cache = runtime_cache.compiler
            self._epilogue_cache = runtime_cache.compiler
            self._kernel_cache = runtime_cache.kernels
        self.fuse_epilogue = self.compile_policy != "off"
        self.diagnostics = diagnostics
        return self

    @property
    def runtime(self) -> SharedBranchRuntime:
        if self._shared_runtime is None:
            self._shared_runtime = SharedBranchRuntime(
                kernel_backend=self.kernel_backend,
                compile_policy=self.compile_policy,
                tile_frames=self.tile_frames,
            )
            self._gather_cache = self._shared_runtime.gather
            self._compiler_cache = self._shared_runtime.compiler
            self._epilogue_cache = self._shared_runtime.compiler
            self._kernel_cache = self._shared_runtime.kernels
        return self._shared_runtime

    def clear(self):
        # Shared caches are owned by VDNState. Standalone instances still release theirs.
        if self._shared_runtime is not None:
            return
        super().clear()

    release = clear

    def _delta_backend(self, length):
        return self.runtime.delta_backend(self.delta_rule, length)

    def _feature_one(
        self, weights, tokens, projection, num_frames, frame_size, *, inference=False, fhsd=False
    ):
        if not inference:
            return super()._feature_one(  # noqa: SLF001
                weights, tokens, projection, num_frames, frame_size,
                inference=False, fhsd=fhsd,
            )
        normalize = projection != "v"
        if projection in self.short_conv:
            if frame_size is None:
                raise ValueError("short convolution requires frame_size")
            gh, gw = frame_size
            heads, head_dim = tokens.shape[-2:]
            channels = heads * head_dim
            spatial_weight = weights[f"short_conv.{projection}_sp.weight"]
            temporal_weight = weights[f"short_conv.{projection}_tm.weight"]
            kh, kw = spatial_weight.shape[-2:]
            volume = tokens.reshape(num_frames, gh, gw, channels).permute(0, 3, 1, 2)
            volume = F.conv2d(
                volume, spatial_weight, padding=(kh // 2, kw // 2), groups=channels
            )
            flattened = volume.permute(0, 2, 3, 1).reshape(num_frames, gh * gw, channels)
            temporal = temporal_weight.squeeze(1).to(flattened.dtype).contiguous()
            return self.runtime.kernels.temporal(
                flattened.contiguous(), temporal, temporal.shape[-1], temporal.shape[-1] // 2,
                heads, head_dim, normalize,
                mode=self.kernel_backend,
                compile_policy=self.compile_policy,
                fhsd=fhsd,
            )
        return self.runtime.kernels.activate(
            tokens, normalize,
            num_frames=num_frames,
            per_frame=(tokens.shape[0] // num_frames if num_frames else None),
            fhsd=fhsd,
            compile_policy=self.compile_policy,
        )

    def _features(self, weights, q_raw, k_raw, v_raw, num_frames, frame_size, *, inference=False):
        if not inference:
            return super()._features(weights, q_raw, k_raw, v_raw, num_frames, frame_size, inference=False)  # noqa: SLF001
        return tuple(
            self._feature_one(
                weights, tensor, projection, num_frames, frame_size,
                inference=True, fhsd=True,
            )
            for projection, tensor in zip(("q", "k", "v"), (q_raw, k_raw, v_raw))
        )

    def _common_statistics(
        self, weights, xv, qkv_raw, num_frames, tokens_per_frame, frame_size, *, inference
    ):
        if not inference:
            return super()._common_statistics(  # noqa: SLF001
                weights, xv, qkv_raw, num_frames, tokens_per_frame, frame_size,
                inference=False,
            )
        with self._scope("features", xv.device):
            query_by_frame, key_by_frame, value_by_frame = self._features(
                weights, *qkv_raw, num_frames, frame_size, inference=True
            )
        beta = torch.sigmoid(F.linear(xv, weights["beta_proj.weight"]))
        beta = beta.reshape(num_frames, tokens_per_frame, self.num_heads).permute(0, 2, 1)
        with self._scope("frame_stats", xv.device):
            matrix_a, matrix_b = reference.frame_statistics(
                key_by_frame, value_by_frame, beta, self.a_fp32,
                inference=True, compiler_cache=self.runtime.compiler,
            )
        frame_mean = xv.reshape(num_frames, tokens_per_frame, -1).mean(dim=1, dtype=torch.float32)
        alpha = reference.alpha_gate(
            frame_mean,
            weights["alpha.down.weight"], weights["alpha.up.weight"],
            weights["alpha.dt_bias"], weights["alpha.A_log"],
            self.num_heads, self.head_dim,
        )
        return query_by_frame, matrix_a, matrix_b, alpha

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
            prefix, suffix = _run_scans_shared(
                self._delta_backend(tokens_per_frame), alpha, matrix_a, matrix_b,
                text_state, self.runtime,
            )
        del matrix_a, matrix_b
        gate = self._output_gate(weights, xv)
        with self._scope("gather", xv.device):
            linear_state = reference.gather_linear_state(
                prefix, suffix, alpha, bounds, bridge=self.bridge, text_state=text_state,
                out_dtype=gate.dtype, cache=self.runtime.gather,
                inference=True, compiler_cache=self.runtime.compiler,
            )
        del prefix, suffix
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        del query_by_frame, linear_state
        with self._scope("epilogue", xv.device):
            return reference.linear_epilogue(
                readout, weights["norm.weight"], gate, 1e-6,
                fuse=self.compile_policy != "off", compiler_cache=self.runtime.compiler,
                inference=self.compile_policy != "off",
            ).reshape(rows, self.num_heads * self.head_dim)

    def _tile_feature(
        self, weights, raw, projection, start, stop, num_frames, tokens_per_frame, frame_size
    ):
        if projection not in self.short_conv:
            rows = slice(start * tokens_per_frame, stop * tokens_per_frame)
            return self._feature_one(
                weights, raw[rows], projection, stop - start, frame_size,
                inference=True, fhsd=True,
            )
        temporal = weights[f"short_conv.{projection}_tm.weight"]
        pad = int(temporal.shape[-1] // 2)
        halo_start = max(0, start - pad)
        halo_stop = min(num_frames, stop + pad)
        rows = slice(halo_start * tokens_per_frame, halo_stop * tokens_per_frame)
        full = self._feature_one(
            weights, raw[rows], projection, halo_stop - halo_start, frame_size,
            inference=True, fhsd=True,
        )
        return full[start - halo_start : stop - halo_start]

    def _statistics_tiled(
        self, weights, xv, k_raw, v_raw, num_frames, tokens_per_frame, frame_size, tile_frames
    ):
        device = xv.device
        matrix_a = torch.empty(
            num_frames, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=torch.float32,
        )
        matrix_b = torch.empty_like(matrix_a)
        alpha = torch.empty(
            num_frames, self.num_heads, self.head_dim, device=device, dtype=torch.float32
        )
        for start in range(0, num_frames, tile_frames):
            stop = min(num_frames, start + tile_frames)
            rows = slice(start * tokens_per_frame, stop * tokens_per_frame)
            x_tile = xv[rows]
            with self._scope("features_kv_tile", device):
                key_tile = self._tile_feature(
                    weights, k_raw, "k", start, stop, num_frames, tokens_per_frame, frame_size
                )
                value_tile = self._tile_feature(
                    weights, v_raw, "v", start, stop, num_frames, tokens_per_frame, frame_size
                )
            beta = torch.sigmoid(F.linear(x_tile, weights["beta_proj.weight"]))
            beta = beta.reshape(stop - start, tokens_per_frame, self.num_heads).permute(0, 2, 1)
            with self._scope("frame_stats_tile", device):
                a_tile, b_tile = reference.frame_statistics(
                    key_tile, value_tile, beta, self.a_fp32,
                    inference=True, compiler_cache=self.runtime.compiler,
                )
            matrix_a[start:stop].copy_(a_tile)
            matrix_b[start:stop].copy_(b_tile)
            frame_mean = x_tile.reshape(stop - start, tokens_per_frame, -1).mean(
                dim=1, dtype=torch.float32
            )
            alpha[start:stop].copy_(
                reference.alpha_gate(
                    frame_mean,
                    weights["alpha.down.weight"], weights["alpha.up.weight"],
                    weights["alpha.dt_bias"], weights["alpha.A_log"],
                    self.num_heads, self.head_dim,
                )
            )
            del key_tile, value_tile, beta, a_tile, b_tile
        return matrix_a, matrix_b, alpha

    def _projected_delta_tiled(
        self,
        weights,
        xv,
        q_raw,
        k_raw,
        v_raw,
        num_frames,
        tokens_per_frame,
        bounds,
        frame_size,
        text_x,
        text_k_raw,
        text_v_raw,
        tile_frames,
        projector,
    ):
        matrix_a, matrix_b, alpha = self._statistics_tiled(
            weights, xv, k_raw, v_raw, num_frames, tokens_per_frame, frame_size, tile_frames
        )
        text_state = self._text_state(weights, text_x, text_k_raw, text_v_raw)
        with self._scope("scan", xv.device):
            prefix, suffix = _run_scans_shared(
                self._delta_backend(tokens_per_frame), alpha, matrix_a, matrix_b,
                text_state, self.runtime,
            )
        del matrix_a, matrix_b
        out_features = int(weights["to_out_linear.weight"].shape[0])
        delta = torch.empty(
            num_frames * tokens_per_frame, out_features, device=xv.device, dtype=xv.dtype
        )
        for start in range(0, num_frames, tile_frames):
            stop = min(num_frames, start + tile_frames)
            rows = slice(start * tokens_per_frame, stop * tokens_per_frame)
            x_tile = xv[rows]
            with self._scope("features_q_tile", xv.device):
                query = self._tile_feature(
                    weights, q_raw, "q", start, stop, num_frames, tokens_per_frame, frame_size
                )
            gate = self._output_gate(weights, x_tile)
            with self._scope("gather_tile", xv.device):
                linear_state = _gather_range(
                    prefix, suffix, alpha, bounds, start, stop,
                    bridge=self.bridge, text_state=text_state, out_dtype=gate.dtype,
                    runtime=self.runtime,
                )
            readout = torch.matmul(query, linear_state.transpose(-1, -2))
            with self._scope("epilogue_tile", xv.device):
                flat = reference.linear_epilogue(
                    readout, weights["norm.weight"], gate, 1e-6,
                    fuse=self.compile_policy != "off", compiler_cache=self.runtime.compiler,
                    inference=self.compile_policy != "off",
                )
            with self._scope("to_out_tile", xv.device):
                delta[rows].copy_(projector(flat.to(dtype=xv.dtype)))
            del query, gate, linear_state, readout, flat
        return delta

    def projected_delta(
        self,
        weights,
        xv,
        q_raw,
        k_raw,
        v_raw,
        num_frames,
        tokens_per_frame,
        bounds,
        *,
        frame_size=None,
        text_x=None,
        text_k_raw=None,
        text_v_raw=None,
        skip_ends=False,
        inference=False,
        projector=None,
        tile_frames: int | None = None,
    ):
        """Return the branch after ``to_out_linear``, projecting only useful rows.

        In inference the tiled spelling first condenses K/V into frame statistics, runs
        the exact bidirectional recurrence, then processes Q/gate/readout/projection in
        frame tiles. Anchor rows are sliced out before all branch work, matching the
        existing trained semantics while avoiding a GEMM over known zeros.
        """
        if projector is None:
            weight = weights["to_out_linear.weight"]
            projector = lambda value: F.linear(value, weight)
        original_rows = num_frames * tokens_per_frame
        row_offset = 0
        if skip_ends:
            if num_frames <= 2:
                return xv.new_zeros(original_rows, int(weights["to_out_linear.weight"].shape[0]))
            row_offset = tokens_per_frame
            inner = slice(row_offset, (num_frames - 1) * tokens_per_frame)
            xv = xv[inner]
            q_raw, k_raw, v_raw = q_raw[inner], k_raw[inner], v_raw[inner]
            num_frames -= 2
            bounds = [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]]

        tile = self.tile_frames if tile_frames is None else int(tile_frames)
        if inference and tile > 0:
            active = self._projected_delta_tiled(
                weights, xv, q_raw, k_raw, v_raw, num_frames, tokens_per_frame,
                bounds, frame_size, text_x, text_k_raw, text_v_raw,
                min(tile, num_frames), projector,
            )
        else:
            readout = self.readout(
                weights, xv, q_raw, k_raw, v_raw, num_frames, tokens_per_frame,
                bounds, frame_size=frame_size, text_x=text_x,
                text_k_raw=text_k_raw, text_v_raw=text_v_raw,
                skip_ends=False, inference=inference,
            )
            active = projector(readout.to(dtype=xv.dtype))

        if not skip_ends:
            return active
        output = active.new_zeros(original_rows, active.shape[-1])
        output[row_offset : row_offset + active.shape[0]] = active
        return output


__all__ = ["OptimizedLinearBranch", "SharedBranchRuntime", "SharedCompilerCache"]
