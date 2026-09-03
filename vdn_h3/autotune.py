"""One-time exact attention autotuning for a VDN render geometry.

Only exact backends installed on the active GPU are considered. The fastest
numerically-valid result is persisted by hardware/PyTorch/geometry, so normal warm
renders pay no benchmarking overhead.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .calibration import calibration_signature


_LOG = logging.getLogger("comfy.vdn_h3")


def _timed(fn, device: torch.device, runs: int = 1):
    # One warm call absorbs lazy kernel/library compilation. The explicit calibration
    # node remains available for longer multi-run measurements.
    warm = fn()
    del warm
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    value = None
    for _ in range(max(1, int(runs))):
        value = fn()
    end.record(torch.cuda.current_stream(device))
    end.synchronize()
    return value, float(start.elapsed_time(end)) / max(1, int(runs))


def _signature(query, layout, cache):
    del cache
    from .window import window_group_count

    return calibration_signature(
        query,
        layout.num_frames,
        layout.bounds,
        layout.anchor_frames,
        groups=window_group_count(layout.num_frames, layout.bounds, layout.anchor_frames),
        video_start=layout.video_start,
        video_end=layout.video_end,
        tokens_per_frame=layout.tokens_per_frame,
    )


def runtime_autotune_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: Any,
    scale: float,
    cache: Any,
    *,
    runs: int = 3,
) -> str | None:
    """Return/persist the fastest exact backend for the current geometry."""

    if not query.is_cuda or not torch.cuda.is_available():
        return None
    signature = _signature(query, layout, cache)
    existing = cache.calibration.lookup(signature)
    if existing:
        cache.last_calibration_hit = existing
        return existing

    from .window import (
        decomposed_available,
        flash2_available,
        flex_available,
        window_softmax_decomposed,
        window_softmax_flash2,
        window_softmax_flex,
        window_softmax_grouped,
    )

    common = dict(
        video_start=layout.video_start,
        video_end=layout.video_end,
        num_frames=layout.num_frames,
        tokens_per_frame=layout.tokens_per_frame,
        bounds=layout.bounds,
        scale=scale,
        anchor_frames=layout.anchor_frames,
    )

    def grouped():
        return window_softmax_grouped(
            query, key, value, **common, transformer_options=None
        )

    candidates: list[tuple[str, Any]] = [("grouped", grouped)]
    if flex_available(cache):
        candidates.append(("flex", lambda: window_softmax_flex(query, key, value, **common, cache=cache)))
    if flash2_available(cache):
        candidates.append(("flash2", lambda: window_softmax_flash2(query, key, value, **common, cache=cache)))
    if decomposed_available(cache):
        candidates.append(("decomposed", lambda: window_softmax_decomposed(query, key, value, **common, cache=cache)))
    if len(candidates) == 1:
        return None

    _LOG.info(
        "VDN-H3 autotuning exact attention for %d frames, %d tokens/frame on %s (%s)",
        layout.num_frames,
        layout.tokens_per_frame,
        torch.cuda.get_device_name(query.device),
        ", ".join(name for name, _ in candidates),
    )
    results: dict[str, dict[str, Any]] = {}
    try:
        reference, grouped_ms = _timed(grouped, query.device, runs=runs)
    except Exception as exc:
        _LOG.warning("VDN-H3 attention autotune could not run grouped reference: %s", exc)
        return None
    results["grouped"] = {
        "available": True,
        "allclose": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "ms": grouped_ms,
    }

    for name, fn in candidates[1:]:
        try:
            got, elapsed = _timed(fn, query.device, runs=runs)
            diff = (got.float() - reference.float()).abs()
            close = bool(torch.allclose(got.float(), reference.float(), atol=2e-2, rtol=4e-2))
            results[name] = {
                "available": True,
                "allclose": close,
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "ms": elapsed,
            }
            del got, diff
        except Exception as exc:
            results[name] = {"available": False, "error": str(exc)}
            cache.mark_broken(name, exc)

    valid = {
        name: float(item["ms"])
        for name, item in results.items()
        if item.get("available") and item.get("allclose") and "ms" in item
    }
    if not valid:
        del reference
        return None
    winner = min(valid, key=valid.get)
    try:
        cache.calibration.record(signature, winner=winner, results=results)
        cache.calibration.save()
        cache.last_calibration_hit = winner
    except Exception as exc:
        _LOG.warning("VDN-H3 could not persist attention calibration: %s", exc)
    finally:
        del reference

    _LOG.info(
        "VDN-H3 attention autotune winner: %s (steady %.3f ms; grouped %.3f ms)",
        winner,
        valid[winner],
        grouped_ms,
    )
    return winner


__all__ = ["runtime_autotune_attention"]
