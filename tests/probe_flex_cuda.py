"""Manual CUDA checks for the Flex window backend; excluded from the pytest suite.

    python <ComfyUI>/custom_nodes/ComfyUI-Kirei-VDN-H3/tests/probe_flex_cuda.py

1. Parity of Flex against the grouped oracle.
2. Twelve distinct packed lengths through one WindowAttentionCache. dynamo must keep the
   compiled kernel (dynamic shapes, tensor-captured mask geometry) instead of hitting its
   recompile limit and falling back to eager flex_attention, which materialises the full
   S x S score matrix. The loop runs with fail_on_recompile_limit_hit so a fallback is an
   exception, and the peak memory of every length is checked against the dense footprint.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PLUGIN_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLUGIN_ROOT))
for _candidate in _PLUGIN_ROOT.parents:
    if (_candidate / "folder_paths.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

import torch  # noqa: E402

from vdn_h3.window import (  # noqa: E402
    WindowAttentionCache,
    recompile_limit,
    window_bounds,
    window_softmax_flex,
    window_softmax_grouped,
)


def parity(device) -> None:
    torch.manual_seed(0)
    frames, per_frame, heads, dim = 12, 8, 2, 128
    video_start = 5
    video_end = video_start + frames * per_frame
    sequence = video_end + 3
    query = torch.randn(sequence, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    bounds = window_bounds(frames, radius=1, chunk=5)
    scale = dim**-0.5
    cache = WindowAttentionCache(limit=2)
    grouped = window_softmax_grouped(
        query, key, value, video_start, video_end, frames, per_frame,
        bounds, scale, anchor_frames="both",
    )
    flex = window_softmax_flex(
        query, key, value, video_start, video_end, frames, per_frame,
        bounds, scale, anchor_frames="both", cache=cache,
    )
    torch.cuda.synchronize()
    error = (grouped.float() - flex.float()).abs().max().item()
    print(f"parity max_abs_error={error}")
    print(f"parity allclose={torch.allclose(grouped.float(), flex.float(), atol=2e-2, rtol=2e-2)}")
    print(f"cache_entries_before_release={len(cache)}")
    cache.release()
    print(f"cache_entries_after_release={len(cache)}")


def many_lengths(device, count: int = 12) -> None:
    import torch._dynamo.config as dynamo_config

    for name in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit"):
        if hasattr(dynamo_config, name):
            setattr(dynamo_config, name, True)
    heads, dim, per_frame = 2, 128, 256
    scale = dim**-0.5
    cache = WindowAttentionCache(limit=32)
    print(f"dynamo recompile limit before compile: {recompile_limit()}")
    for index in range(count):
        frames = 7 + 2 * index
        video_start = 5 + index
        video_end = video_start + frames * per_frame
        sequence = video_end + 3 + index
        query = torch.randn(sequence, heads, dim, device=device, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        bounds = window_bounds(frames, radius=1, chunk=5)
        out = window_softmax_flex(
            query, key, value, video_start, video_end, frames, per_frame,
            bounds, scale, anchor_frames="both", cache=cache,
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        out = window_softmax_flex(
            query, key, value, video_start, video_end, frames, per_frame,
            bounds, scale, anchor_frames="both", cache=cache,
        )
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - before
        dense = sequence * sequence * heads * 4  # fp32 S x S scores: the eager footprint
        reference = window_softmax_grouped(
            query, key, value, video_start, video_end, frames, per_frame,
            bounds, scale, anchor_frames="both",
        )
        close = torch.allclose(out.float(), reference.float(), atol=2e-2, rtol=2e-2)
        print(
            f"length {index + 1:2d}: S={sequence:6d} peak={peak / 2**20:8.1f} MiB "
            f"dense_scores={dense / 2**20:8.1f} MiB allclose={close}"
        )
        if not close:
            raise SystemExit(f"FAIL: parity lost at length {sequence}")
        if peak >= dense:
            raise SystemExit(f"FAIL: peak memory at length {sequence} matches the eager S x S footprint")
    print(f"dynamo recompile limit after {count} lengths: {recompile_limit()}")
    print("PASS")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    parity(device)
    many_lengths(device)


if __name__ == "__main__":
    main()
