"""Independent CPU tests for the packed window-attention implementation."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vdn_h3_private import window


def _dense_oracle(q, k, v, video_start, video_end, frames, per_frame, bounds, scale, anchors):
    sequence = q.shape[0]
    query_rows = torch.arange(sequence).view(-1, 1)
    key_rows = torch.arange(sequence).view(1, -1)
    global_query = (query_rows < video_start) | (query_rows >= video_end)
    global_key = (key_rows < video_start) | (key_rows >= video_end)
    query_frame = torch.div(query_rows - video_start, per_frame, rounding_mode="floor")
    key_frame = torch.div(key_rows - video_start, per_frame, rounding_mode="floor")

    lower = torch.zeros(sequence, dtype=torch.long)
    upper = torch.full((sequence,), frames - 1, dtype=torch.long)
    for frame, (lo, hi) in enumerate(bounds):
        start = video_start + frame * per_frame
        lower[start:start + per_frame] = max(lo, 0)
        upper[start:start + per_frame] = min(hi, frames - 1)
    allowed = global_query | global_key | (
        (key_frame >= lower[:, None]) & (key_frame <= upper[:, None])
    )
    if anchors in ("columns", "both"):
        allowed |= (key_frame == 0) | (key_frame == frames - 1)
    if anchors in ("rows", "both"):
        allowed |= (query_frame == 0) | (query_frame == frames - 1)
    return F.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        attn_mask=allowed.unsqueeze(0).unsqueeze(0),
        scale=scale,
    ).squeeze(0).permute(1, 0, 2)


@pytest.mark.parametrize("chunk,radius", [(0, 2), (3, 1), (4, 2)])
@pytest.mark.parametrize("anchors", window.ANCHOR_FRAME_MODES)
def test_grouped_attention_matches_dense_partition(chunk, radius, anchors):
    generator = torch.Generator().manual_seed(1000 + chunk * 10 + radius)
    frames, per_frame, heads, dim = 11, 3, 2, 5
    before, after = 4, 2
    video_start = before
    video_end = before + frames * per_frame
    sequence = video_end + after
    q = torch.randn(sequence, heads, dim, generator=generator)
    k = torch.randn(sequence, heads, dim, generator=generator)
    v = torch.randn(sequence, heads, dim, generator=generator)
    bounds = window.window_bounds(frames, radius, chunk)
    scale = dim**-0.5
    got = window.window_softmax_grouped(
        q, k, v, video_start, video_end, frames, per_frame,
        bounds, scale, anchor_frames=anchors,
    )
    want = _dense_oracle(
        q, k, v, video_start, video_end, frames, per_frame,
        bounds, scale, anchors,
    )
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)


def test_grouped_video_queries_are_contiguous_slices(monkeypatch):
    frames, per_frame, heads, dim = 12, 2, 2, 4
    video_start, video_end = 3, 3 + frames * per_frame
    q = torch.randn(video_end + 2, heads, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    calls = []
    original = window._sdpa

    def recording_sdpa(q_rows, *args, **kwargs):
        calls.append(q_rows)
        return original(q_rows, *args, **kwargs)

    monkeypatch.setattr(window, "_sdpa", recording_sdpa)
    window.window_softmax_grouped(
        q, k, v, video_start, video_end, frames, per_frame,
        window.window_bounds(frames, 1, 4), dim**-0.5,
    )
    # First call is the concatenated global query. Every video call thereafter
    # shares q's storage and spans whole adjacent-frame runs.
    assert len(calls) == 4
    for group in calls[1:]:
        assert group.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
        assert group.is_contiguous()
        assert group.shape[0] % per_frame == 0


def test_comfy_dispatch_contract(monkeypatch):
    calls = []

    def optimized(q, k, v, heads, mask=None, **kwargs):
        calls.append((q.shape, k.shape, heads, kwargs["transformer_options"]))
        assert mask is None
        head_dim = q.shape[-1] // heads
        result = F.scaled_dot_product_attention(
            q.view(1, q.shape[1], heads, head_dim).transpose(1, 2),
            k.view(1, k.shape[1], heads, head_dim).transpose(1, 2),
            v.view(1, v.shape[1], heads, head_dim).transpose(1, 2),
            scale=head_dim**-0.5,
        )
        return result.transpose(1, 2).reshape_as(q)

    modules = {}
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules"):
        modules[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    attention_module = types.ModuleType("comfy.ldm.modules.attention")
    attention_module.optimized_attention = optimized
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", attention_module)

    frames, per_frame, heads, dim = 7, 2, 2, 4
    video_start, video_end = 2, 2 + frames * per_frame
    q = torch.randn(video_end + 1, heads, dim)
    k, v = torch.randn_like(q), torch.randn_like(q)
    options = {"patches": "preserved"}
    got = window.window_softmax_grouped(
        q, k, v, video_start, video_end, frames, per_frame,
        window.window_bounds(frames, 1, 3), dim**-0.5,
        transformer_options=options,
    )
    want = window.window_softmax_grouped(
        q, k, v, video_start, video_end, frames, per_frame,
        window.window_bounds(frames, 1, 3), dim**-0.5,
    )
    torch.testing.assert_close(got, want)
    assert calls and all(call[2] == heads and call[3] is options for call in calls)


def test_window_geometry_validation_and_coverage():
    assert window.window_bounds(4, 1) == [(-1, 1), (0, 2), (1, 3), (2, 4)]
    assert window.window_bounds(6, 1, 5)[0] == (-5, 9)
    assert window.window_bounds(6, 1, 5)[5] == (0, 14)
    assert window.full_coverage(window.window_bounds(6, 1, 5), 6)
    assert not window.full_coverage(window.window_bounds(16, 1, 5), 16)
    with pytest.raises(ValueError):
        window.window_bounds(3, -1)
    with pytest.raises(ValueError):
        window.full_coverage([(0, 1)], 2)


def test_block_mask_cache_is_model_owned_bounded_and_releasable():
    assert window._device_key(torch.device("cuda:0")) != window._device_key(torch.device("cuda:1"))
    cache = window.WindowAttentionCache(limit=7)
    for index in range(40):
        cache.put((index,), index)
    assert list(cache._masks) == [(index,) for index in range(33, 40)]
    cache.release()
    assert len(cache) == 0 and cache._compiled is None
    assert not hasattr(window, "_BLOCK_MASK_CACHE")
