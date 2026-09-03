"""Runtime report and explicit attention calibration nodes for Kirei VDN-H3."""

from __future__ import annotations

import json

import torch

from .benchmark import runtime_snapshot


class KireiVDNH3RuntimeReport:
    """Expose resolved VDN runtime as JSON while passing the MODEL through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "after": (
                    "*",
                    {
                        "tooltip": (
                            "Optional dependency trigger. Connect the sampler LATENT or another "
                            "downstream output here to capture metrics after rendering."
                        )
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report_json")
    FUNCTION = "report"
    CATEGORY = "model_patches/video/advanced"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Return selected VDN-H3 memory, kernel, attention, FP8 and diagnostics state. "
        "Connect a sampler output to 'after' for a post-render report."
    )

    def report(self, model, after=None):
        del after
        snapshot = runtime_snapshot(model)
        return model, json.dumps(snapshot, indent=2, sort_keys=True, default=str)


def _state(model):
    value = getattr(model, "object_patches", {}).get("diffusion_model._vdn_h3_state")
    if value is None:
        raise RuntimeError("MODEL does not carry Kirei VDN-H3 state")
    return value


def _timed_cuda(fn, device, runs):
    for _ in range(1):
        value = fn()
        del value
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    value = None
    for _ in range(runs):
        value = fn()
    end.record(torch.cuda.current_stream(device))
    end.synchronize()
    return value, float(start.elapsed_time(end)) / runs


class KireiVDNH3CalibrateAttention:
    """Benchmark exact attention backends for one explicit GPU/geometry and persist winner."""

    @classmethod
    def INPUT_TYPES(cls):
        advanced = {"advanced": True}
        return {
            "required": {
                "model": ("MODEL",),
                "num_frames": ("INT", {"default": 41, "min": 3, "max": 256, "step": 1}),
                "tokens_per_frame": (
                    "INT", {"default": 384, "min": 1, "max": 4096, "step": 1}
                ),
                "global_tokens": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 1}),
            },
            "optional": {
                "runs": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, **advanced}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1, **advanced}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "calibration_json")
    FUNCTION = "calibrate"
    CATEGORY = "model_patches/video/advanced"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Explicitly benchmark grouped SDPA, FlexAttention, FA2 and FA4 for a synthetic "
        "geometry, verify numerical parity, then persist the fastest exact backend. "
        "Normal generation never runs this benchmark automatically."
    )

    def calibrate(self, model, num_frames, tokens_per_frame, global_tokens, runs=3, seed=1234):
        state = _state(model)
        device = torch.device(getattr(model, "load_device", "cpu"))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("attention calibration requires the patched H3 MODEL on CUDA")
        from .calibration import calibration_signature
        from .window import (
            decomposed_available,
            flash2_available,
            flex_available,
            window_bounds,
            window_group_count,
            window_softmax_decomposed,
            window_softmax_flash2,
            window_softmax_flex,
            window_softmax_grouped,
        )

        torch.manual_seed(int(seed))
        frames = int(num_frames)
        per_frame = int(tokens_per_frame)
        video_start = int(global_tokens)
        video_end = video_start + frames * per_frame
        sequence = video_end
        heads, dim = int(state.num_heads), int(state.head_dim)
        dtype = torch.bfloat16
        q = torch.randn(sequence, heads, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        bounds = window_bounds(
            frames,
            int(state.config.get("radius", 1)),
            int(state.config.get("chunk", 5)),
        )
        anchors = str(state.config.get("anchor_frames", "both"))
        scale = dim**-0.5
        cache = state.window_cache

        def grouped():
            return window_softmax_grouped(
                q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
                anchor_frames=anchors, transformer_options=None,
            )

        reference, grouped_ms = _timed_cuda(grouped, device, int(runs))
        results = {
            "grouped": {
                "available": True,
                "allclose": True,
                "max_abs": 0.0,
                "ms": grouped_ms,
            }
        }
        candidates = []
        if flex_available(cache):
            candidates.append(
                (
                    "flex",
                    lambda: window_softmax_flex(
                        q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
                        anchor_frames=anchors, cache=cache,
                    ),
                )
            )
        if flash2_available(cache):
            candidates.append(
                (
                    "flash2",
                    lambda: window_softmax_flash2(
                        q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
                        anchor_frames=anchors, cache=cache,
                    ),
                )
            )
        if decomposed_available(cache):
            candidates.append(
                (
                    "decomposed",
                    lambda: window_softmax_decomposed(
                        q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
                        anchor_frames=anchors, cache=cache,
                    ),
                )
            )

        for name, fn in candidates:
            try:
                got, elapsed = _timed_cuda(fn, device, int(runs))
                diff = (got.float() - reference.float()).abs()
                close = bool(
                    torch.allclose(got.float(), reference.float(), atol=2e-2, rtol=4e-2)
                )
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

        valid = {
            name: item["ms"]
            for name, item in results.items()
            if item.get("available") and item.get("allclose") and "ms" in item
        }
        if not valid:
            raise RuntimeError("no numerically valid attention backend completed calibration")
        winner = min(valid, key=valid.get)
        signature = calibration_signature(
            q,
            frames,
            bounds,
            anchors,
            groups=window_group_count(frames, bounds, anchors),
            video_start=video_start,
            video_end=video_end,
            tokens_per_frame=per_frame,
        )
        cache.calibration.record(signature, winner=winner, results=results)
        cache.calibration.save()
        payload = {
            "winner": winner,
            "gpu": torch.cuda.get_device_name(device),
            "sequence": sequence,
            "video_start": video_start,
            "video_end": video_end,
            "frames": frames,
            "tokens_per_frame": per_frame,
            "global_tokens": int(global_tokens),
            "heads": heads,
            "head_dim": dim,
            "results": results,
            "calibration": cache.calibration.snapshot(),
        }
        del reference, q, k, v
        return model, json.dumps(payload, indent=2, sort_keys=True)


NODE_CLASS_MAPPINGS = {
    "KireiVDNH3RuntimeReport": KireiVDNH3RuntimeReport,
    "KireiVDNH3CalibrateAttention": KireiVDNH3CalibrateAttention,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiVDNH3RuntimeReport": "Kirei VDN-H3 Runtime Report",
    "KireiVDNH3CalibrateAttention": "Kirei VDN-H3 Calibrate Attention",
}


__all__ = [
    "KireiVDNH3RuntimeReport",
    "KireiVDNH3CalibrateAttention",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
