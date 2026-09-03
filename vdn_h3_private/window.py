"""Window geometry and grouped softmax for the private VDN-H3 port.

The mathematical semantics are based on the Apache-2.0 OpenVDN implementation
and its Saganaki22 ComfyUI integration (see ``THIRD_PARTY.md``).  This module is
an independent, dependency-light implementation intended to be usable in CPU
tests as well as inside ComfyUI.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")


class WindowAttentionCache:
    """Model-owned LRU for FlexAttention masks and its compiled callable.

    Block masks contain device tensors.  Keeping this object on ``VDNState``
    gives those allocations the same lifetime as the model instead of retaining
    them in module globals after unload.
    """

    def __init__(self, limit: int = 16):
        if limit <= 0:
            raise ValueError("cache limit must be positive")
        self.limit = int(limit)
        self._masks: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._compiled = None

    def get(self, key: tuple[Any, ...]):
        value = self._masks.get(key)
        if value is not None:
            self._masks.move_to_end(key)
        return value

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        _cache_put(self._masks, key, value, self.limit)

    def attention(self, flex_attention):
        if self._compiled is None:
            self._compiled = torch.compile(flex_attention)
        return self._compiled

    def clear(self) -> None:
        self._masks.clear()
        self._compiled = None

    def release(self) -> None:
        self.clear()

    def __len__(self) -> int:
        return len(self._masks)


def window_bounds(num_frames: int, radius: int, chunk: int = 0) -> list[tuple[int, int]]:
    """Return inclusive, unclamped frame bounds for every query frame.

    ``chunk <= 0`` selects a frame-centred window.  Otherwise the window is
    aligned to groups of ``chunk`` frames; every frame in one chunk therefore
    has the same bounds.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if isinstance(chunk, bool) or not isinstance(chunk, int):
        raise ValueError("chunk must be an integer")
    if chunk <= 0:
        return [(frame - radius, frame + radius) for frame in range(num_frames)]
    return [
        (
            ((frame // chunk) - radius) * chunk,
            ((frame // chunk) + radius + 1) * chunk - 1,
        )
        for frame in range(num_frames)
    ]


def full_coverage(bounds: Sequence[tuple[int, int]], num_frames: int) -> bool:
    """Whether all query windows already contain the complete video."""
    if len(bounds) != num_frames:
        raise ValueError("bounds must contain exactly one entry per frame")
    return all(lo <= 0 and hi >= num_frames - 1 for lo, hi in bounds)


def _validate_layout(
    sequence_length: int,
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: Sequence[tuple[int, int]],
    anchor_frames: str,
) -> None:
    if anchor_frames not in ANCHOR_FRAME_MODES:
        raise ValueError(
            f"anchor_frames={anchor_frames!r}; expected one of {ANCHOR_FRAME_MODES}"
        )
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


def _sdpa(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    scale: float | None,
    transformer_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run dense attention over row-major ``[S, H, D]`` tensors."""
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

    result = F.scaled_dot_product_attention(
        q_rows.permute(1, 0, 2).unsqueeze(0),
        k_rows.permute(1, 0, 2).unsqueeze(0),
        v_rows.permute(1, 0, 2).unsqueeze(0),
        scale=scale,
    )
    return result.squeeze(0).permute(1, 0, 2)


def _contiguous_runs(frames: Sequence[int]) -> list[tuple[int, int]]:
    """Convert ordered frame numbers to inclusive/exclusive contiguous runs."""
    if not frames:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            runs.append((start, previous + 1))
            start = frame
        previous = frame
    runs.append((start, previous + 1))
    return runs


def _select_keys(
    key: torch.Tensor,
    value: torch.Tensor,
    video_start: int,
    video_end: int,
    tokens_per_frame: int,
    frames: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather globals and video-frame runs, slicing only contiguous row ranges."""
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    if video_start:
        key_parts.append(key[:video_start])
        value_parts.append(value[:video_start])
    if video_end < key.shape[0]:
        key_parts.append(key[video_end:])
        value_parts.append(value[video_end:])
    for first, stop in _contiguous_runs(frames):
        row_start = video_start + first * tokens_per_frame
        row_stop = video_start + stop * tokens_per_frame
        key_parts.append(key[row_start:row_stop])
        value_parts.append(value[row_start:row_stop])
    # A valid video always contributes at least the current frame.
    if len(key_parts) == 1:
        return key_parts[0], value_parts[0]
    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def window_softmax_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: Sequence[tuple[int, int]],
    scale: float | None,
    anchor_frames: str = "none",
    transformer_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Exact packed-sequence window attention using contiguous query groups.

    All pairs involving a global token remain dense.  Video-to-video pairs are
    limited to the frame bounds, with optional dense anchor rows/columns.  A run
    of adjacent frames with identical effective key frames is processed by one
    SDPA call and direct slice assignment; no per-token index tensor is built.
    """
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key and value must share shape [sequence, heads, dim]")
    _validate_layout(
        query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
        bounds, anchor_frames,
    )

    output = torch.empty_like(query)
    # There are at most two contiguous global regions.  Concatenation is needed
    # only for their dense query call and never creates frame-by-frame fragments.
    global_q_parts = []
    global_positions = []
    if video_start:
        global_q_parts.append(query[:video_start])
        global_positions.append(slice(0, video_start))
    if video_end < query.shape[0]:
        global_q_parts.append(query[video_end:])
        global_positions.append(slice(video_end, query.shape[0]))
    if global_q_parts:
        global_queries = (
            global_q_parts[0] if len(global_q_parts) == 1
            else torch.cat(global_q_parts, dim=0)
        )
        dense = _sdpa(global_queries, key, value, scale, transformer_options)
        offset = 0
        for rows in global_positions:
            count = rows.stop - rows.start
            output[rows] = dense[offset:offset + count]
            offset += count

    anchor_columns = anchor_frames in ("columns", "both")
    anchor_rows = anchor_frames in ("rows", "both")
    last_frame = num_frames - 1

    # Make maximal adjacent runs with the same key-frame tuple.  This is more
    # general than grouping by a dict: even custom bounds never turn a group into
    # a non-contiguous advanced-index gather.
    runs: list[tuple[int, int, tuple[int, ...] | None]] = []
    run_start = 0
    previous_signature: tuple[int, ...] | None | object = object()
    for frame in range(num_frames):
        dense_row = anchor_rows and frame in (0, last_frame)
        if dense_row:
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

    for first_frame, stop_frame, selected_frames in runs:
        row_start = video_start + first_frame * tokens_per_frame
        row_stop = video_start + stop_frame * tokens_per_frame
        if selected_frames is None:
            selected_key, selected_value = key, value
        else:
            selected_key, selected_value = _select_keys(
                key, value, video_start, video_end, tokens_per_frame, selected_frames
            )
        output[row_start:row_stop] = _sdpa(
            query[row_start:row_stop], selected_key, selected_value,
            scale, transformer_options,
        )
    return output


def _build_window_tables(
    sequence_length: int,
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: Sequence[tuple[int, int]],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-query-token clamped video-frame bounds for FlexAttention."""
    _validate_layout(
        sequence_length, video_start, video_end, num_frames, tokens_per_frame,
        bounds, "none",
    )
    lo = torch.zeros(sequence_length, dtype=torch.long, device=device)
    hi = torch.full((sequence_length,), num_frames - 1, dtype=torch.long, device=device)
    for frame, (lower, upper) in enumerate(bounds):
        row_start = video_start + frame * tokens_per_frame
        row_stop = row_start + tokens_per_frame
        lo[row_start:row_stop] = max(lower, 0)
        hi[row_start:row_stop] = min(upper, num_frames - 1)
    return lo, hi


def _window_mask_mod(
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    lo: torch.Tensor,
    hi: torch.Tensor,
    anchor_frames: str,
):
    """Create the scalar mask predicate consumed by FlexAttention."""
    if anchor_frames not in ANCHOR_FRAME_MODES:
        raise ValueError(f"unknown anchor frame mode: {anchor_frames!r}")
    allow_anchor_columns = anchor_frames in ("columns", "both")
    allow_anchor_rows = anchor_frames in ("rows", "both")

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        global_query = (query_index < video_start) | (query_index >= video_end)
        global_key = (key_index < video_start) | (key_index >= video_end)
        query_frame = (query_index - video_start) // tokens_per_frame
        key_frame = (key_index - video_start) // tokens_per_frame
        allowed = global_query | global_key | (
            (key_frame >= lo[query_index]) & (key_frame <= hi[query_index])
        )
        if allow_anchor_columns:
            allowed = allowed | (key_frame == 0) | (key_frame == num_frames - 1)
        if allow_anchor_rows:
            allowed = allowed | (query_frame == 0) | (query_frame == num_frames - 1)
        return allowed

    return mask_mod


def _device_key(device: torch.device) -> tuple[str, int | None]:
    resolved = torch.device(device)
    return resolved.type, resolved.index


def _cache_put(cache: OrderedDict, key: tuple[Any, ...], value: Any, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def window_softmax_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: Sequence[tuple[int, int]],
    scale: float | None,
    anchor_frames: str = "none",
    cache: WindowAttentionCache | None = None,
) -> torch.Tensor:
    """Execute the same partition with a cached FlexAttention BlockMask.

    When a model-owned ``cache`` is supplied it is LRU-bounded and keys CUDA
    device indices separately.  With the default ``None`` no mask or compiled
    callable is retained by this module.
    """
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key and value must share shape [sequence, heads, dim]")
    _validate_layout(
        query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
        bounds, anchor_frames,
    )
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    attention = cache.attention(flex_attention) if cache is not None else flex_attention
    cache_key = (
        query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
        anchor_frames, tuple(tuple(pair) for pair in bounds), _device_key(query.device),
    )
    block_mask = cache.get(cache_key) if cache is not None else None
    if block_mask is None:
        lo, hi = _build_window_tables(
            query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
            bounds, query.device,
        )
        block_mask = create_block_mask(
            _window_mask_mod(
                video_start, video_end, num_frames, tokens_per_frame,
                lo, hi, anchor_frames,
            ),
            None, None, query.shape[0], query.shape[0], query.device,
            _compile=True,
        )
        if cache is not None:
            cache.put(cache_key, block_mask)
    output = attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask,
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1)


__all__ = [
    "ANCHOR_FRAME_MODES",
    "WindowAttentionCache",
    "full_coverage",
    "window_bounds",
    "window_softmax_flex",
    "window_softmax_grouped",
]
