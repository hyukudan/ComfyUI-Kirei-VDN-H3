"""Manual CUDA smoke test; excluded from the normal pytest suite."""

from __future__ import annotations

import torch

from vdn_h3.window import (
    WindowAttentionCache,
    window_bounds,
    window_softmax_flex,
    window_softmax_grouped,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
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
    print(torch.cuda.get_device_name(0))
    print(f"max_abs_error={error}")
    print(f"allclose={torch.allclose(grouped.float(), flex.float(), atol=2e-2, rtol=2e-2)}")
    print(f"cache_entries_before_release={len(cache)}")
    cache.release()
    print(f"cache_entries_after_release={len(cache)}")


if __name__ == "__main__":
    main()
