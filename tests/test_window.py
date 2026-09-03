"""Independent CPU tests for the packed window-attention implementation."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vdn_h3 import window


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
    assert len(calls) == 4
    for group in calls[1:]:
        assert group.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
        assert group.is_contiguous()
        assert group.shape[0] % per_frame == 0


def test_exact_sdpa_uses_comfy_backend_priority_without_override(monkeypatch):
    calls = []
    comfy_module = types.ModuleType("comfy")
    ops_module = types.ModuleType("comfy.ops")

    def exact_sdpa(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs.get("scale")))
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    ops_module.scaled_dot_product_attention = exact_sdpa
    comfy_module.ops = ops_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops_module)

    rows, heads, dim = 9, 2, 4
    q = torch.randn(rows, heads, dim)
    k = torch.randn(rows + 3, heads, dim)
    v = torch.randn_like(k)
    scale = dim**-0.5
    got = window._sdpa(q, k, v, scale, transformer_options=None)
    want = F.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        scale=scale,
    ).squeeze(0).permute(1, 0, 2)
    torch.testing.assert_close(got, want)
    assert calls == [((1, heads, rows, dim), (1, heads, rows + 3, dim), scale)]


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


def test_gpu_family_never_treats_consumer_blackwell_as_datacenter(monkeypatch):
    cases = (
        ((9, 0), "hopper", True),
        ((10, 0), "blackwell_dc", True),
        ((10, 3), "blackwell_dc", True),
        ((12, 0), "blackwell_consumer", False),
        ((8, 9), "ada", False),
        ((8, 6), "ampere", False),
    )
    for capability, family, fa4 in cases:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device, c=capability: c)
        assert window.gpu_family("cuda:0") == family
        assert window.prefers_fa4("cuda:0") is fa4


def test_grouped_copy_bytes_grow_with_global_rows():
    frames, per_frame, heads, dim = 10, 4, 2, 8
    bounds = window.window_bounds(frames, 1, 5)
    per_row = 2 * heads * dim * 2  # K and V, bf16

    def measure(globals_rows):
        seq = globals_rows + frames * per_frame
        q = torch.zeros(seq, heads, dim, dtype=torch.bfloat16)
        return window.grouped_copy_bytes(
            q, frames, per_frame, bounds, "both", globals_rows, seq
        )

    peak_small, total_small = measure(2)
    peak_big, total_big = measure(1000)
    # frames 1..8 share one window covering all ten frames; frames 0 and 9 are dense rows
    assert peak_small == total_small == (2 + frames * per_frame) * per_row
    assert peak_big == total_big == (1000 + frames * per_frame) * per_row


def test_grouped_copy_guard_reasons(monkeypatch):
    gib = 1024**3
    assert window.grouped_copies_too_large("cuda:0", 1 * gib, 9 * gib)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10 * gib, 96 * gib))
    assert window.grouped_copies_too_large("cuda:0", 4 * gib, 4 * gib)
    assert window.grouped_copies_too_large("cuda:0", 1 * gib, 2 * gib) is None


def test_fa4_kernel_generation_follows_the_family(monkeypatch):
    cases = {
        (10, 0): ("tcgen05", True),
        (9, 0): ("wgmma", True),
        (12, 0): ("mma_sync", False),
        (8, 9): ("mma_sync", False),
        (8, 6): ("mma_sync", False),
        (7, 5): ("unsupported", False),
    }
    for capability, (kernel, first) in cases.items():
        monkeypatch.setattr(
            window.torch.cuda, "get_device_capability", lambda device=None, c=capability: c
        )
        assert window.fa4_kernel("cuda:0") == kernel
        assert window.prefers_fa4("cuda:0") is first


def test_flex_wrapper_compiles_static_and_raises_the_recompile_floor(monkeypatch):
    import torch._dynamo.config as dynamo_config

    captured = {}

    def fake_compile(fn, **kwargs):
        captured.update(kwargs)
        return fn

    monkeypatch.setattr(window.torch, "compile", fake_compile)
    for name in ("recompile_limit", "cache_size_limit"):
        monkeypatch.setattr(dynamo_config, name, 8, raising=False)
    cache = window.WindowAttentionCache()

    def sentinel(*args, **kwargs):
        return None

    assert cache.attention(sentinel) is sentinel
    assert captured == {"dynamic": False}
    assert window.recompile_limit() >= window.DYNAMO_RECOMPILE_FLOOR
    for name in ("recompile_limit", "cache_size_limit"):
        monkeypatch.setattr(dynamo_config, name, 512, raising=False)
    assert window.raise_recompile_limit() == 512  # never lowered


def test_mask_mod_handles_global_and_anchor_geometry():
    lo = torch.zeros(40, dtype=torch.long)
    hi = torch.full((40,), 2, dtype=torch.long)
    hi[24:34] = 0  # frame 2 only sees frame 0
    mask_mod = window._window_mask_mod(4, 34, 3, 10, lo, hi, "none")
    q, k = torch.tensor(30), torch.tensor
    assert bool(mask_mod(0, 0, q, k(18))) is False  # frame 1 is outside frame 2's window
    assert bool(mask_mod(0, 0, q, k(8))) is True  # frame 0 inside
    assert bool(mask_mod(0, 0, q, k(36))) is True  # global key
    assert bool(mask_mod(0, 0, torch.tensor(1), k(18))) is True  # global query
    anchored = window._window_mask_mod(4, 34, 3, 10, lo, hi, "both")
    assert bool(anchored(0, 0, q, k(18))) is True  # last frame is an anchor row


def test_dispatch_reason_is_rewritten_on_every_resolution(tmp_path):
    q = torch.zeros(20, 2, 4)
    bounds = window.window_bounds(3, 1)
    cache = window.WindowAttentionCache()
    cache.calibration = window.CalibrationStore(tmp_path / "cal.json")
    cache.last_dispatch_reason = "stale"
    assert window.resolve_attention_backend("flex", q, 3, bounds, "both", cache) == "flex"
    assert cache.last_dispatch_reason == "explicit: flex"
    geometry = dict(video_start=2, video_end=14, tokens_per_frame=4)
    assert window.resolve_attention_backend("auto", q, 3, bounds, "both", cache, **geometry) == "grouped"
    assert cache.last_dispatch_reason.startswith("heuristic: grouped")
    signature = window.calibration_signature(
        q, 3, bounds, "both", groups=window.window_group_count(3, bounds, "both"), **geometry
    )
    cache.calibration.record(signature, winner="grouped", results={})
    assert window.resolve_attention_backend("auto", q, 3, bounds, "both", cache, **geometry) == "grouped"
    assert cache.last_dispatch_reason == "calibrated: grouped"
    assert cache.last_calibration_hit == "grouped"
