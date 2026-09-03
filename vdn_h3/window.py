"""Window-attention backends for VDN-H3: grouped, Flex and Blackwell decomposition."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F


ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")
ATTENTION_BACKENDS = ("auto", "grouped", "flex", "decomposed", "reference")


def _cache_put(cache: OrderedDict, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


class WindowAttentionCache:
    """All device masks/plans/compiled calls belong to one patched VDN model."""

    def __init__(self, limit: int = 16):
        if limit <= 0:
            raise ValueError("cache limit must be positive")
        self.limit = int(limit)
        self._masks: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._plans: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._compiled = None
        self._broken: dict[str, str] = {}

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
                self._compiled = torch.compile(flex_attention, dynamic=False)
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
    result = F.scaled_dot_product_attention(
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

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        global_query = (query_index < video_start) | (query_index >= video_end)
        global_key = (key_index < video_start) | (key_index >= video_end)
        query_frame = (query_index - video_start) // tokens_per_frame
        key_frame = (key_index - video_start) // tokens_per_frame
        allowed = global_query | global_key | ((key_frame >= lo[query_index]) & (key_frame <= hi[query_index]))
        if allow_anchor_columns:
            allowed = allowed | (key_frame == 0) | (key_frame == num_frames - 1)
        if allow_anchor_rows:
            allowed = allowed | (query_frame == 0) | (query_frame == num_frames - 1)
        return allowed

    return mask_mod


def _device_key(device):
    resolved = torch.device(device)
    return resolved.type, resolved.index


def flex_available(cache: WindowAttentionCache | None = None) -> bool:
    if cache is not None and cache.is_broken("flex"):
        return False
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa:F401
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


def resolve_attention_backend(
    requested: str,
    query: torch.Tensor,
    num_frames: int,
    bounds,
    anchor_frames: str,
    cache: WindowAttentionCache | None = None,
) -> str:
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(f"unsupported attention backend {requested!r}")
    if requested != "auto":
        return requested
    if query.is_cuda and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(query.device)
        if capability[0] >= 10 and decomposed_available(cache):
            return "decomposed"
    groups = window_group_count(num_frames, bounds, anchor_frames)
    if groups <= 8:
        return "grouped"
    if query.is_cuda and query.shape[0] >= 8192 and flex_available(cache):
        return "flex"
    return "grouped"


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


def window_softmax_decomposed(
    query, key, value, video_start, video_end, num_frames, tokens_per_frame,
    bounds, scale, anchor_frames="none", cache: WindowAttentionCache | None = None,
):
    if not query.is_cuda:
        raise RuntimeError("decomposed FA4 attention requires CUDA")
    from flash_attn.cute.interface import flash_attn_varlen_func

    plan = _decomposed_plan(
        cache, query.shape[0], video_start, video_end, num_frames, tokens_per_frame,
        bounds, anchor_frames, query.device,
    )
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    out = torch.empty_like(query)
    if len(plan.dense_q):
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
    if plan.has_windows:
        kw = key[plan.kv_gather]
        vw = value[plan.kv_gather]
        ow = flash_attn_varlen_func(
            query[plan.win_q], kw, vw,
            cu_seqlens_q=plan.cu_q, cu_seqlens_k=plan.cu_k,
            max_seqlen_q=plan.max_q, max_seqlen_k=plan.max_k,
            softmax_scale=scale,
        )
        if isinstance(ow, tuple):
            ow = ow[0]
        out[plan.win_q] = ow
    return out


__all__ = [
    "ANCHOR_FRAME_MODES",
    "ATTENTION_BACKENDS",
    "WindowAttentionCache",
    "decomposed_available",
    "flex_available",
    "full_coverage",
    "resolve_attention_backend",
    "window_bounds",
    "window_group_count",
    "window_softmax_decomposed",
    "window_softmax_flex",
    "window_softmax_grouped",
]
