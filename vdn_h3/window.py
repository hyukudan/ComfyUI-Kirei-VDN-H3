"""Window-attention backends for VDN-H3: grouped, Flex, FA2 and FA4 (CuTe DSL)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from .calibration import CalibrationStore, calibration_signature


_LOG = logging.getLogger("comfy.vdn_h3")
ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")
ATTENTION_BACKENDS = (
    "auto",
    "grouped",
    "flex",
    "flash2",
    "decomposed",
    "reference",
    "compat",
)


DYNAMO_RECOMPILE_FLOOR = 64
_RECOMPILE_LIMIT_NAMES = ("recompile_limit", "cache_size_limit")


def _dynamo_config():
    import torch._dynamo.config as dynamo_config

    return dynamo_config


def recompile_limit() -> int | None:
    """Current dynamo per-function recompile limit (``None`` without dynamo)."""
    try:
        config = _dynamo_config()
    except Exception:
        return None
    values = [int(getattr(config, name)) for name in _RECOMPILE_LIMIT_NAMES if hasattr(config, name)]
    return min(values) if values else None


def raise_recompile_limit(floor: int = DYNAMO_RECOMPILE_FLOOR) -> int | None:
    """Raise dynamo's recompile limit to at least ``floor``; never lower it.

    Past the limit dynamo stops compiling a function and runs it eagerly. For
    ``flex_attention`` and ``create_block_mask`` eager means a dense S x S intermediate,
    so that fallback has to stay out of reach however many layouts a session sees.
    """
    try:
        config = _dynamo_config()
    except Exception:
        return None
    current = recompile_limit()
    if current is not None and current < floor:
        for name in _RECOMPILE_LIMIT_NAMES:
            if hasattr(config, name):
                setattr(config, name, int(floor))
    return recompile_limit()


def _cache_put(cache: OrderedDict, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


class WindowAttentionCache:
    """All masks/plans/compiled calls and calibration belong to one patched model."""

    def __init__(self, limit: int = 16):
        if limit <= 0:
            raise ValueError("cache limit must be positive")
        self.limit = int(limit)
        self._masks: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._plans: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._compiled = None
        self._broken: dict[str, str] = {}
        self.calibration = CalibrationStore()
        self.last_calibration_hit: str | None = None
        self.last_autotune_error: str | None = None
        self.last_dispatch_reason: str | None = None

    def get(self, key):
        value = self._masks.get(key)
        if value is not None:
            self._masks.move_to_end(key)
        return value

    def put(self, key, value):
        _cache_put(self._masks, key, value, self.limit)

    def get_plan(self, key):
        value = self._plans.get(key)
        if value is not None:
            self._plans.move_to_end(key)
        return value

    def put_plan(self, key, value):
        _cache_put(self._plans, key, value, min(self.limit, 8))

    def attention(self, flex_attention):
        if self._compiled is None and not self.is_broken("flex_compile"):
            try:
                raise_recompile_limit()
                # dynamic=True: one compiled kernel serves every packed length. A static
                # compile recompiles per length and, past dynamo's recompile limit,
                # silently falls back to eager flex_attention, which materialises the
                # full S x S score matrix (extreme slowdown or OOM on long clips).
                self._compiled = torch.compile(flex_attention, dynamic=True)
            except Exception as exc:
                self.mark_broken("flex_compile", exc)
        return self._compiled or flex_attention

    def mark_broken(self, backend: str, reason: Any):
        self._broken[str(backend)] = str(reason)

    def is_broken(self, backend: str) -> bool:
        return str(backend) in self._broken

    def clear(self):
        self._masks.clear()
        self._plans.clear()
        self._compiled = None
        self._broken.clear()
        self.last_calibration_hit = None
        self.last_autotune_error = None
        self.last_dispatch_reason = None

    release = clear

    def __len__(self):
        return len(self._masks)


def window_bounds(num_frames: int, radius: int, chunk: int = 0) -> list[tuple[int, int]]:
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if isinstance(chunk, bool) or not isinstance(chunk, int):
        raise ValueError("chunk must be an integer")
    if chunk <= 0:
        return [(frame - radius, frame + radius) for frame in range(num_frames)]
    return [
        (((frame // chunk) - radius) * chunk, ((frame // chunk) + radius + 1) * chunk - 1)
        for frame in range(num_frames)
    ]


def full_coverage(bounds, num_frames):
    if len(bounds) != num_frames:
        raise ValueError("bounds must contain exactly one entry per frame")
    return all(lo <= 0 and hi >= num_frames - 1 for lo, hi in bounds)


def _validate_layout(sequence_length, video_start, video_end, num_frames, tokens_per_frame, bounds, anchor_frames):
    if anchor_frames not in ANCHOR_FRAME_MODES:
        raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {ANCHOR_FRAME_MODES}")
    if num_frames <= 0 or tokens_per_frame <= 0:
        raise ValueError("num_frames and tokens_per_frame must be positive")
    if not 0 <= video_start <= video_end <= sequence_length:
        raise ValueError("video range is outside the packed sequence")
    if video_end - video_start != num_frames * tokens_per_frame:
        raise ValueError("video range does not equal num_frames * tokens_per_frame")
    if len(bounds) != num_frames:
        raise ValueError("bounds must contain exactly one entry per frame")
    for frame, pair in enumerate(bounds):
        if len(pair) != 2:
            raise ValueError(f"bounds[{frame}] must be a (lo, hi) pair")
        lo, hi = pair
        if lo > frame or hi < frame or lo > hi:
            raise ValueError(f"bounds[{frame}]={pair!r} does not contain its query frame")


def _exact_sdpa(q, k, v, *, scale):
    """Exact SDPA with Comfy's backend-priority dispatch when available.

    This deliberately does *not* route through ``optimized_attention`` or any model
    ``transformer_options`` override, so Sage/kitchen quantized attention cannot soften
    VDN's trained local windows.  On current ComfyUI, however, ``comfy.ops`` still
    selects the fastest exact PyTorch backend for the platform (Flash/cuDNN/efficient),
    which matters especially on Windows builds.
    """
    try:
        from comfy.ops import scaled_dot_product_attention as sdpa
    except ImportError:
        sdpa = F.scaled_dot_product_attention
    return sdpa(q, k, v, scale=scale)


def _sdpa(q_rows, k_rows, v_rows, scale, transformer_options=None):
    if transformer_options is not None:
        from comfy.ldm.modules import attention as comfy_attention

        rows, heads, head_dim = q_rows.shape
        result = comfy_attention.optimized_attention(
            q_rows.reshape(1, rows, heads * head_dim),
            k_rows.reshape(1, k_rows.shape[0], heads * head_dim),
            v_rows.reshape(1, v_rows.shape[0], heads * head_dim),
            heads,
            transformer_options=transformer_options,
        )
        return result.reshape(rows, heads, head_dim)
    result = _exact_sdpa(
        q_rows.permute(1, 0, 2).unsqueeze(0),
        k_rows.permute(1, 0, 2).unsqueeze(0),
        v_rows.permute(1, 0, 2).unsqueeze(0),
        scale=scale,
    )
    return result.squeeze(0).permute(1, 0, 2)


def _contiguous_runs(frames: Sequence[int]):
    if not frames:
        return []
    runs = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            runs.append((start, previous + 1))
            start = frame
        previous = frame
    runs.append((start, previous + 1))
    return runs


def _select_keys(key, value, video_start, video_end, tokens_per_frame, frames):
    key_parts, value_parts = [], []
    if video_start:
        key_parts.append(key[:video_start]); value_parts.append(value[:video_start])
    if video_end < key.shape[0]:
        key_parts.append(key[video_end:]); value_parts.append(value[video_end:])
    for first, stop in _contiguous_runs(frames):
        a = video_start + first * tokens_per_frame
        b = video_start + stop * tokens_per_frame
        key_parts.append(key[a:b]); value_parts.append(value[a:b])
    if len(key_parts) == 1:
        return key_parts[0], value_parts[0]
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def _window_runs(num_frames, bounds, anchor_frames):
    anchor_columns = anchor_frames in ("columns", "both")
    anchor_rows = anchor_frames in ("rows", "both")
    last_frame = num_frames - 1
    runs = []
    run_start = 0
    previous_signature = object()
    for frame in range(num_frames):
        if anchor_rows and frame in (0, last_frame):
            signature = None
        else:
            lo = max(bounds[frame][0], 0)
            hi = min(bounds[frame][1], last_frame)
            selected = set(range(lo, hi + 1))
            if anchor_columns:
                selected.update((0, last_frame))
            signature = tuple(sorted(selected))
        if frame and signature != previous_signature:
            runs.append((run_start, frame, previous_signature))
            run_start = frame
        previous_signature = signature
    runs.append((run_start, num_frames, previous_signature))
    return runs


def window_group_count(num_frames, bounds, anchor_frames="none") -> int:
    return len(_window_runs(num_frames, bounds, anchor_frames))


def window_softmax_grouped(
    query, key, value, video_start, video_end, num_frames, tokens_per_frame,
    bounds, scale, anchor_frames="none", transformer_options=None,
):
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key and value must share shape [sequence, heads, dim]")
    _validate_layout(query.shape[0], video_start, video_end, num_frames, tokens_per_frame, bounds, anchor_frames)
    output = torch.empty_like(query)
    global_q_parts, global_positions = [], []
    if video_start:
        global_q_parts.append(query[:video_start]); global_positions.append(slice(0, video_start))
    if video_end < query.shape[0]:
        global_q_parts.append(query[video_end:]); global_positions.append(slice(video_end, query.shape[0]))
    if global_q_parts:
        global_queries = global_q_parts[0] if len(global_q_parts) == 1 else torch.cat(global_q_parts, dim=0)
        dense = _sdpa(global_queries, key, value, scale, transformer_options)
        offset = 0
        for rows in global_positions:
            count = rows.stop - rows.start
            output[rows] = dense[offset : offset + count]
            offset += count
    for first_frame, stop_frame, selected_frames in _window_runs(num_frames, bounds, anchor_frames):
        row_start = video_start + first_frame * tokens_per_frame
        row_stop = video_start + stop_frame * tokens_per_frame
        if selected_frames is None:
            selected_key, selected_value = key, value
        else:
            selected_key, selected_value = _select_keys(
                key, value, video_start, video_end, tokens_per_frame, selected_frames
            )
        output[row_start:row_stop] = _sdpa(
            query[row_start:row_stop], selected_key, selected_value, scale, transformer_options
        )
    return output


def _build_window_tables(sequence_length, video_start, video_end, num_frames, tokens_per_frame, bounds, device):
    _validate_layout(sequence_length, video_start, video_end, num_frames, tokens_per_frame, bounds, "none")
    lo = torch.zeros(sequence_length, dtype=torch.long, device=device)
    hi = torch.full((sequence_length,), num_frames - 1, dtype=torch.long, device=device)
    for frame, (lower, upper) in enumerate(bounds):
        a = video_start + frame * tokens_per_frame
        b = a + tokens_per_frame
        lo[a:b] = max(lower, 0); hi[a:b] = min(upper, num_frames - 1)
    return lo, hi


def _window_mask_mod(video_start, video_end, num_frames, tokens_per_frame, lo, hi, anchor_frames):
    allow_anchor_columns = anchor_frames in ("columns", "both")
    allow_anchor_rows = anchor_frames in ("rows", "both")
    # The geometry is captured as 0-d tensors, not Python ints. torch.compile guards on
    # the value of every captured int, so each new layout would add a cache entry to the
    # compiled create_block_mask body until dynamo hit its recompile limit and ran the
    # eager path, which materialises the full S x S mask. Tensors are guarded by
    # shape/dtype only.
    geometry = torch.tensor(
        [int(video_start), int(video_end), int(num_frames) - 1, int(tokens_per_frame)],
        dtype=torch.long,
        device=lo.device,
    )
    v_start, v_end, last_frame, per_frame = geometry[0], geometry[1], geometry[2], geometry[3]

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        global_query = (query_index < v_start) | (query_index >= v_end)
        global_key = (key_index < v_start) | (key_index >= v_end)
        query_frame = (query_index - v_start) // per_frame
        key_frame = (key_index - v_start) // per_frame
        allowed = global_query | global_key | ((key_frame >= lo[query_index]) & (key_frame <= hi[query_index]))
        if allow_anchor_columns:
            allowed = allowed | (key_frame == 0) | (key_frame == last_frame)
        if allow_anchor_rows:
            allowed = allowed | (query_frame == 0) | (query_frame == last_frame)
        return allowed

    return mask_mod


def _device_key(device):
    resolved = torch.device(device)
    return resolved.type, resolved.index


def gpu_family(device) -> str:
    """Coarse NVIDIA family for kernel policy.

    Datacenter Hopper (sm_90) and Blackwell (sm_100/sm_103) run the wgmma / tcgen05
    FlashAttention-4 kernels and the per-tensor FP8 cuBLAS path OpenVDN tuned for.
    Consumer and workstation Blackwell (sm_120, RTX 50xx / RTX PRO 6000) shares the name
    but not those kernels: flash-attn-4 runs its mma.sync (SM80-class) kernel there, so
    it must never be treated as sm_100.
    """
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return "unknown"
    if major == 12:
        return "blackwell_consumer"
    if major == 10:
        return "blackwell_dc"
    if major == 9:
        return "hopper"
    if major == 8:
        return "ada" if minor >= 9 else "ampere"
    return f"sm_{major}{minor}"


FA4_KERNELS = {
    "blackwell_dc": "tcgen05",
    "hopper": "wgmma",
    "blackwell_consumer": "mma_sync",
    "ada": "mma_sync",
    "ampere": "mma_sync",
}


def fa4_kernel(device) -> str:
    """Which flash-attn-4 (CuTe DSL) kernel generation this device would run.

    ``tcgen05`` (sm_100) and ``wgmma`` (sm_90) are the generations OpenVDN measured.
    ``mma_sync`` is the SM80-class kernel flash-attn-4 also builds for sm_120, Ada and
    Ampere; whether it beats grouped SDPA or Flex there is a calibration result.
    """
    family = gpu_family(device)
    if family == "unknown":
        return "unknown"
    return FA4_KERNELS.get(family, "unsupported")


def prefers_fa4(device) -> bool:
    """Whether the decomposed FA4/CuTe window kernel is worth trying before calibrating.

    Only the tcgen05 / wgmma generations skip the queue; the mma.sync generation competes
    in the calibration like grouped, Flex and FA2.
    """
    return fa4_kernel(device) in {"tcgen05", "wgmma"}


GROUPED_COPY_GUARD_FRACTION = 0.35
GROUPED_COPY_GUARD_TOTAL_BYTES = 8 * 1024**3


def grouped_copy_bytes(
    query, num_frames, tokens_per_frame, bounds, anchor_frames, video_start, video_end
) -> tuple[int, int]:
    """(peak, total) bytes of the K/V copies ``window_softmax_grouped`` makes per layer.

    Every window group concatenates the global rows (text, audio, keyframes, reference
    video) with its window frames. With a long reference clip those globals dominate:
    the peak decides whether a group fits, the total decides how much bandwidth the
    copies burn compared with a block-sparse Flex kernel that reads K/V in place.
    """
    global_rows = int(video_start) + (int(query.shape[0]) - int(video_end))
    per_row = 2 * int(query.shape[1]) * int(query.shape[2]) * int(query.element_size())
    peak = total = 0
    for _first, _stop, selected in _window_runs(num_frames, bounds, anchor_frames):
        if selected is None:
            continue  # dense anchor rows reuse the full K/V without copying
        size = (global_rows + len(selected) * int(tokens_per_frame)) * per_row
        peak = max(peak, size)
        total += size
    return peak, total


def grouped_copies_too_large(device, peak_bytes: int, total_bytes: int) -> str | None:
    """Reason string when grouped attention should be skipped for this layout."""
    if total_bytes > GROUPED_COPY_GUARD_TOTAL_BYTES:
        return f"grouped K/V copies would move {total_bytes / 2**30:.1f} GiB per layer"
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    if peak_bytes > GROUPED_COPY_GUARD_FRACTION * free:
        return (
            f"one grouped K/V copy ({peak_bytes / 2**30:.1f} GiB) exceeds "
            f"{GROUPED_COPY_GUARD_FRACTION:.0%} of free VRAM"
        )
    return None


def flex_available(cache: WindowAttentionCache | None = None) -> bool:
    if cache is not None and cache.is_broken("flex"):
        return False
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa:F401
        return True
    except Exception:
        return False


def flash2_available(cache: WindowAttentionCache | None = None) -> bool:
    if cache is not None and cache.is_broken("flash2"):
        return False
    try:
        from flash_attn import flash_attn_varlen_func  # noqa:F401
        return True
    except Exception:
        return False


def decomposed_available(cache: WindowAttentionCache | None = None) -> bool:
    if cache is not None and cache.is_broken("decomposed"):
        return False
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func  # noqa:F401
        return True
    except Exception:
        return False


def backend_inventory(cache: WindowAttentionCache | None = None) -> dict[str, bool]:
    """Which exact window backends can run here (pure import checks when ``cache`` is None)."""
    return {
        "grouped": True,
        "flex": flex_available(cache),
        "flash2": flash2_available(cache),
        "decomposed": decomposed_available(cache),
    }


def _backend_available(backend: str, cache: WindowAttentionCache | None):
    if backend == "grouped":
        return True
    if backend == "flex":
        return flex_available(cache)
    if backend == "flash2":
        return flash2_available(cache)
    if backend == "decomposed":
        return decomposed_available(cache)
    return False


def _autotune_if_needed(
    query,
    num_frames,
    bounds,
    anchor_frames,
    cache,
    video_start,
    video_end,
    tokens_per_frame,
):
    if (
        cache is None
        or not query.is_cuda
        or not torch.cuda.is_available()
        or video_start is None
        or video_end is None
        or tokens_per_frame is None
    ):
        return None
    available = 1
    available += int(flex_available(cache))
    available += int(flash2_available(cache))
    available += int(decomposed_available(cache))
    if available <= 1:
        return None
    layout = SimpleNamespace(
        num_frames=int(num_frames),
        bounds=tuple(tuple(pair) for pair in bounds),
        anchor_frames=str(anchor_frames),
        video_start=int(video_start),
        video_end=int(video_end),
        tokens_per_frame=int(tokens_per_frame),
    )
    try:
        from .autotune import runtime_autotune_attention

        # Runtime is shape-driven. Reusing the real Q tensor for Q/K/V avoids allocating
        # three synthetic inputs while preserving the exact backend geometry.
        return runtime_autotune_attention(
            query,
            query,
            query,
            layout,
            query.shape[-1] ** -0.5,
            cache,
            runs=2,
        )
    except Exception as exc:
        cache.last_autotune_error = str(exc)
        _LOG.warning("VDN-H3 attention autotune failed; using conservative dispatch: %s", exc)
        return None


def _heuristic_backend(query, groups, cache) -> tuple[str, str]:
    """Conservative choice when nothing was calibrated; returns (backend, why)."""
    if query.is_cuda and torch.cuda.is_available():
        if prefers_fa4(query.device) and decomposed_available(cache):
            return "decomposed", f"{fa4_kernel(query.device)} FA4 kernel installed"
    if groups <= 8:
        return "grouped", f"{groups} window groups"
    if query.is_cuda and query.shape[0] >= 8192 and flex_available(cache):
        return "flex", f"{groups} window groups, {int(query.shape[0])} tokens"
    if not query.is_cuda:
        return "grouped", f"{groups} window groups, cpu"
    return "grouped", f"{groups} window groups, flex unavailable or short sequence"


def resolve_attention_backend(
    requested: str,
    query: torch.Tensor,
    num_frames: int,
    bounds,
    anchor_frames: str,
    cache: WindowAttentionCache | None = None,
    *,
    video_start: int | None = None,
    video_end: int | None = None,
    tokens_per_frame: int | None = None,
) -> str:
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(f"unsupported attention backend {requested!r}")
    if cache is not None:
        # Every resolution rewrites the reason; a stale one must never outlive a hit.
        cache.last_dispatch_reason = None
    if requested != "auto":
        if cache is not None:
            cache.last_dispatch_reason = f"explicit: {requested}"
        return requested
    groups = window_group_count(num_frames, bounds, anchor_frames)
    signature = None
    if cache is not None:
        signature = calibration_signature(
            query,
            num_frames,
            bounds,
            anchor_frames,
            groups=groups,
            video_start=video_start,
            video_end=video_end,
            tokens_per_frame=tokens_per_frame,
        )
        calibrated = cache.calibration.lookup(signature)
        if calibrated and _backend_available(calibrated, cache):
            cache.last_calibration_hit = calibrated
            cache.last_dispatch_reason = f"calibrated: {calibrated}"
            return calibrated
        cache.last_calibration_hit = None
        if (
            query.is_cuda
            and video_start is not None
            and video_end is not None
            and tokens_per_frame is not None
            and flex_available(cache)
        ):
            peak, total = grouped_copy_bytes(
                query, num_frames, tokens_per_frame, bounds, anchor_frames, video_start, video_end
            )
            reason = grouped_copies_too_large(query.device, peak, total)
            if reason is not None:
                # Many global rows (reference video, keyframes): grouped would copy them
                # for every window and the autotune itself could OOM. Flex reads K/V in
                # place through the block mask, so it is chosen without benchmarking.
                cache.last_dispatch_reason = f"flex: {reason}"
                return "flex"
        tuned = _autotune_if_needed(
            query,
            num_frames,
            bounds,
            anchor_frames,
            cache,
            video_start,
            video_end,
            tokens_per_frame,
        )
        if tuned and _backend_available(tuned, cache):
            cache.last_calibration_hit = tuned
            cache.last_dispatch_reason = f"autotuned: {tuned}"
            return tuned
    # CPU or a runtime with only grouped attention keeps the conservative heuristics.
    backend, why = _heuristic_backend(query, groups, cache)
    if cache is not None:
        cache.last_dispatch_reason = f"heuristic: {backend} ({why})"
    return backend


def window_softmax_flex(
    query, key, value, video_start, video_end, num_frames, tokens_per_frame,
    bounds, scale, anchor_frames="none", cache: WindowAttentionCache | None = None,
):
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key and value must share shape [sequence, heads, dim]")
    _validate_layout(query.shape[0], video_start, video_end, num_frames, tokens_per_frame, bounds, anchor_frames)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    attention = cache.attention(flex_attention) if cache is not None else flex_attention
    cache_key = (
        query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
        anchor_frames, tuple(tuple(pair) for pair in bounds), _device_key(query.device),
    )
    block_mask = cache.get(cache_key) if cache is not None else None
    if block_mask is None:
        lo, hi = _build_window_tables(
            query.shape[0], video_start, video_end, num_frames, tokens_per_frame, bounds, query.device
        )
        block_mask = create_block_mask(
            _window_mask_mod(video_start, video_end, num_frames, tokens_per_frame, lo, hi, anchor_frames),
            None, None, query.shape[0], query.shape[0], query.device, _compile=True,
        )
        if cache is not None:
            cache.put(cache_key, block_mask)
    output = attention(
        query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0), block_mask=block_mask, scale=scale,
    )
    return output.squeeze(0).transpose(0, 1)


class _DecomposedPlan:
    def __init__(self, sequence_length, video_start, video_end, num_frames, tokens_per_frame, bounds, anchor_frames, device):
        S, F_, TPF = sequence_length, num_frames, tokens_per_frame
        anchor_set = {0, F_ - 1} if anchor_frames in ("columns", "rows", "both") else set()
        dense_rows = anchor_set if anchor_frames in ("rows", "both") else set()
        dense_cols = anchor_set if anchor_frames in ("columns", "both") else set()

        def frame_rows(frame):
            return video_start + frame * TPF, video_start + (frame + 1) * TPF

        globals_ = [r for r in ((0, video_start), (video_end, S)) if r[0] < r[1]]

        def merge(ranges):
            out = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges):
            if not ranges:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.cat([torch.arange(a, b, dtype=torch.long, device=device) for a, b in ranges])

        self.dense_q = cat_ranges(merge(globals_ + [frame_rows(f) for f in sorted(dense_rows)]))
        groups = []
        for f in range(F_):
            if f in dense_rows:
                continue
            if groups and bounds[groups[-1][-1]] == bounds[f] and groups[-1][-1] == f - 1:
                groups[-1].append(f)
            else:
                groups.append([f])
        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(max(lo, 0), min(hi + 1, F_))) | dense_cols)
            qi = cat_ranges(merge([frame_rows(f) for f in frames]))
            ki = cat_ranges(merge(globals_ + [frame_rows(f) for f in kv_frames]))
            q_idx.append(qi); kv_idx.append(ki); q_lens.append(len(qi)); k_lens.append(len(ki))
        self.has_windows = bool(groups)
        if self.has_windows:
            self.win_q = torch.cat(q_idx); self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(device, torch.int32)
            self.cu_k = torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(device, torch.int32)
            self.max_q, self.max_k = max(q_lens), max(k_lens)
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)
            self.kv_gather = self.cu_q = self.cu_k = None
            self.max_q = self.max_k = 0
        order = torch.cat([self.dense_q, self.win_q])
        if len(order) != S:
            raise ValueError(f"decomposition covers {len(order)} of {S} rows")


def _decomposed_plan(cache, sequence_length, video_start, video_end, num_frames, tokens_per_frame, bounds, anchor_frames, device):
    key = (
        sequence_length, video_start, video_end, num_frames, tokens_per_frame,
        tuple(tuple(x) for x in bounds), anchor_frames, _device_key(device),
    )
    plan = cache.get_plan(key) if cache is not None else None
    if plan is None:
        plan = _DecomposedPlan(
            sequence_length, video_start, video_end, num_frames, tokens_per_frame,
            bounds, anchor_frames, device,
        )
        if cache is not None:
            cache.put_plan(key, plan)
    return plan


def _dense_decomposed_rows(query, key, value, plan, scale, out):
    if not len(plan.dense_q):
        return
    qd = query[plan.dense_q]
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
    except Exception:
        context = nullcontext()
    with context:
        od = F.scaled_dot_product_attention(
            qd.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0), scale=scale,
        )
    out[plan.dense_q] = od[0].transpose(0, 1)


def _window_softmax_varlen(
    query, key, value, video_start, video_end, num_frames, tokens_per_frame,
    bounds, scale, anchor_frames, cache, flash_func,
):
    if not query.is_cuda:
        raise RuntimeError("varlen flash attention requires CUDA")
    plan = _decomposed_plan(
        cache, query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
        bounds, anchor_frames, query.device,
    )
    key = key if key.is_contiguous() else key.contiguous()
    value = value if value.is_contiguous() else value.contiguous()
    out = torch.empty_like(query)
    _dense_decomposed_rows(query, key, value, plan, scale, out)
    if plan.has_windows:
        kw = key[plan.kv_gather]
        vw = value[plan.kv_gather]
        ow = flash_func(
            query[plan.win_q], kw, vw,
            cu_seqlens_q=plan.cu_q, cu_seqlens_k=plan.cu_k,
            max_seqlen_q=plan.max_q, max_seqlen_k=plan.max_k,
            softmax_scale=scale,
        )
        if isinstance(ow, tuple):
            ow = ow[0]
        out[plan.win_q] = ow
    return out


def window_softmax_decomposed(
    query, key, value, video_start, video_end, num_frames, tokens_per_frame,
    bounds, scale, anchor_frames="none", cache: WindowAttentionCache | None = None,
):
    from flash_attn.cute.interface import flash_attn_varlen_func

    return _window_softmax_varlen(
        query, key, value, video_start, video_end, num_frames, tokens_per_frame,
        bounds, scale, anchor_frames, cache, flash_attn_varlen_func,
    )


def window_softmax_flash2(
    query, key, value, video_start, video_end, num_frames, tokens_per_frame,
    bounds, scale, anchor_frames="none", cache: WindowAttentionCache | None = None,
):
    """FA2 varlen decomposition for Ada/Ampere-class cards; opt-in or calibrated."""
    from flash_attn import flash_attn_varlen_func

    return _window_softmax_varlen(
        query, key, value, video_start, video_end, num_frames, tokens_per_frame,
        bounds, scale, anchor_frames, cache, flash_attn_varlen_func,
    )


__all__ = [
    "ANCHOR_FRAME_MODES",
    "ATTENTION_BACKENDS",
    "DYNAMO_RECOMPILE_FLOOR",
    "GROUPED_COPY_GUARD_FRACTION",
    "GROUPED_COPY_GUARD_TOTAL_BYTES",
    "WindowAttentionCache",
    "backend_inventory",
    "decomposed_available",
    "flash2_available",
    "flex_available",
    "full_coverage",
    "gpu_family",
    "grouped_copies_too_large",
    "grouped_copy_bytes",
    "fa4_kernel",
    "prefers_fa4",
    "raise_recompile_limit",
    "recompile_limit",
    "resolve_attention_backend",
    "window_bounds",
    "window_group_count",
    "window_softmax_decomposed",
    "window_softmax_flash2",
    "window_softmax_flex",
    "window_softmax_grouped",
]
