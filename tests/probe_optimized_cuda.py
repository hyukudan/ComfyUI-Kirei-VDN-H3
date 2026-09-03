"""Standalone CUDA validation/benchmark for the optimized VDN-H3 runtime.

Run inside the same Python environment used by ComfyUI, for example:

    python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-6000.json
    python tests/probe_optimized_cuda.py --device cuda:1 --json vdn-4090.json

The probe uses synthetic tensors only. It validates numerical parity for the accelerated
kernels/backends, exercises double-buffered branch streaming and emits machine-readable
results suitable for comparing GPUs or future commits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vdn_h3 import branch, window
from vdn_h3.kernels import (
    LinearKernelCache,
    activate,
    temporal_conv1d,
    temporal_shift_eager,
    triton_temporal_conv_activate,
)
from vdn_h3.weights import ManagedBranchWeights


def _sync(device):
    torch.cuda.synchronize(device)


def _timed(device, fn, warmup=2, iters=8):
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _error(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
        "allclose": bool(torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=4e-2)),
    }


def probe_temporal(device, quick=False):
    torch.manual_seed(100)
    frames = 17 if quick else 33
    spatial = 96 if quick else 384
    heads, head_dim = 8, 64
    channels = heads * head_dim
    x = torch.randn(frames, spatial, channels, device=device, dtype=torch.bfloat16).contiguous()
    w = (torch.randn(channels, 5, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    eager_raw = temporal_shift_eager(x, w, 5, 2)
    reference = activate(eager_raw.reshape(-1, heads, head_dim), True)

    out = {
        "shape": list(x.shape),
        "reference_ms": _timed(
            device,
            lambda: activate(temporal_shift_eager(x, w, 5, 2).reshape(-1, heads, head_dim), True),
            iters=3 if quick else 6,
        ),
    }

    conv = lambda: activate(temporal_conv1d(x, w, 5, 2).reshape(-1, heads, head_dim), True)
    conv_value = conv()
    out["conv1d"] = {
        **_error(conv_value, reference),
        "ms": _timed(device, conv, iters=4 if quick else 10),
    }

    cache = LinearKernelCache()
    compiled = lambda: cache.temporal(x, w, 5, 2, heads, head_dim, True, mode="compile")
    compiled_value = compiled()
    out["compile"] = {
        **_error(compiled_value, reference),
        "ms": _timed(device, compiled, warmup=1, iters=4 if quick else 10),
        "compiled_graphs": len(cache._compiled),
        "broken_shapes": len(cache._broken),
    }

    try:
        triton_value = triton_temporal_conv_activate(x, w, 5, 2, heads, head_dim, True)
        triton_fn = lambda: triton_temporal_conv_activate(x, w, 5, 2, heads, head_dim, True)
        out["triton"] = {
            "available": True,
            **_error(triton_value, reference),
            "ms": _timed(device, triton_fn, warmup=2, iters=5 if quick else 15),
        }
    except Exception as exc:
        out["triton"] = {"available": False, "error": str(exc)}
    cache.release()
    return out


def probe_scan(device, quick=False):
    torch.manual_seed(101)
    frames, heads, dim, tokens = (9, 4, 32, 24) if quick else (33, 8, 64, 64)
    key = torch.randn(frames, heads, tokens, dim, device=device, dtype=torch.bfloat16)
    beta = torch.rand(frames, heads, tokens, device=device, dtype=torch.float32)
    a, b = branch.frame_statistics(key, torch.randn_like(key), beta, True)
    alpha = torch.rand(frames, heads, dim, device=device, dtype=torch.float32) * 0.3 + 0.65
    backend = branch.VdnDelta(tokens)
    reference = branch.run_scans_reference(backend, alpha, a, b)
    with torch.no_grad():
        inference = branch.run_scans_inference(backend, alpha, a, b)
    parity = [_error(got, want) for got, want in zip(inference, reference)]
    with torch.no_grad():
        ms = _timed(
            device,
            lambda: branch.run_scans_inference(backend, alpha, a, b),
            warmup=1,
            iters=3 if quick else 8,
        )
    return {"shape": [frames, heads, dim, dim], "directions": parity, "inference_ms": ms}


def probe_attention(device, quick=False):
    torch.manual_seed(102)
    frames = 17 if quick else 33
    per_frame = 16 if quick else 64
    global_before, global_after = 24, 8
    video_start = global_before
    video_end = video_start + frames * per_frame
    sequence = video_end + global_after
    heads, dim = 4, 64
    q = torch.randn(sequence, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bounds = window.window_bounds(frames, 1, 5)
    scale = dim ** -0.5
    cache = window.WindowAttentionCache(limit=8)

    grouped_fn = lambda: window.window_softmax_grouped(
        q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
        anchor_frames="both",
    )
    grouped = grouped_fn()
    result = {
        "shape": list(q.shape),
        "groups": window.window_group_count(frames, bounds, "both"),
        "auto_resolved": window.resolve_attention_backend("auto", q, frames, bounds, "both", cache),
        "grouped_ms": _timed(device, grouped_fn, iters=3 if quick else 8),
    }

    if window.flex_available(cache):
        try:
            flex_fn = lambda: window.window_softmax_flex(
                q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
                anchor_frames="both", cache=cache,
            )
            flex_value = flex_fn()
            result["flex"] = {
                "available": True,
                **_error(flex_value, grouped),
                "ms": _timed(device, flex_fn, warmup=1, iters=3 if quick else 8),
            }
        except Exception as exc:
            result["flex"] = {"available": False, "error": str(exc)}
    else:
        result["flex"] = {"available": False}

    if window.decomposed_available(cache):
        try:
            dec_fn = lambda: window.window_softmax_decomposed(
                q, k, v, video_start, video_end, frames, per_frame, bounds, scale,
                anchor_frames="both", cache=cache,
            )
            dec_value = dec_fn()
            result["decomposed"] = {
                "available": True,
                **_error(dec_value, grouped),
                "ms": _timed(device, dec_fn, warmup=1, iters=3 if quick else 8),
            }
        except Exception as exc:
            result["decomposed"] = {"available": False, "error": str(exc)}
    else:
        result["decomposed"] = {"available": False}
    result["backend_failures"] = dict(cache._broken)
    cache.release()
    return result


def probe_streaming(device, quick=False, stream_mib=None):
    if stream_mib is None:
        stream_mib = 8 if quick else 64
    elements = max(1024, stream_mib * 1024 * 1024 // 4)
    blocks = []
    for index in range(4):
        blocks.append(
            {
                "payload": torch.full((elements,), float(index + 1), dtype=torch.float32),
                "alpha.A_log": torch.full((8,), float(index), dtype=torch.float32),
                "alpha.dt_bias": torch.full((64,), float(index), dtype=torch.float32),
            }
        )
    store = ManagedBranchWeights(blocks, mode="stream")
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    checks = []
    for index in range(len(blocks)):
        _sync(device)
        start = time.perf_counter()
        weights = store.weights_on(index, device, torch.bfloat16)
        observed = weights["payload"][:1024].float().mean()
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
        checks.append(
            {
                "block": index,
                "mean": float(observed.item()),
                "expected": float(index + 1),
                "a_log_fp32": weights["alpha.A_log"].dtype == torch.float32,
                "dt_bias_fp32": weights["alpha.dt_bias"].dtype == torch.float32,
            }
        )
        store.mark_consumed(index, device, torch.bfloat16)
    peak = int(torch.cuda.max_memory_allocated(device))
    store.release()
    _sync(device)
    return {
        "source_mib_per_block": stream_mib,
        "transfer_ms": timings,
        "mean_transfer_ms": sum(timings) / len(timings),
        "checks": checks,
        "peak_allocated_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stream-mib", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    result = {
        "torch": torch.__version__,
        "device": str(device),
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_bytes": int(total),
        "free_bytes_start": int(free),
        "quick": bool(args.quick),
    }

    for name, fn in (
        ("temporal", lambda: probe_temporal(device, args.quick)),
        ("scan", lambda: probe_scan(device, args.quick)),
        ("attention", lambda: probe_attention(device, args.quick)),
        ("streaming", lambda: probe_streaming(device, args.quick, args.stream_mib)),
    ):
        try:
            result[name] = fn()
        except Exception as exc:
            result[name] = {"fatal_error": str(exc)}

    free_end, _ = torch.cuda.mem_get_info(device)
    result["free_bytes_end"] = int(free_end)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        path = Path(args.json_path)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")

    failures = []
    for section in ("temporal", "scan", "attention", "streaming"):
        if "fatal_error" in result.get(section, {}):
            failures.append(section)
    for mode in ("conv1d", "compile", "triton"):
        item = result.get("temporal", {}).get(mode, {})
        if item.get("available", True) and item.get("allclose") is False:
            failures.append(f"temporal.{mode}")
    for mode in ("flex", "decomposed"):
        item = result.get("attention", {}).get(mode, {})
        if item.get("available") and item.get("allclose") is False:
            failures.append(f"attention.{mode}")
    if any(not item.get("allclose", False) for item in result.get("scan", {}).get("directions", [])):
        failures.append("scan.parity")
    if any(
        abs(item["mean"] - item["expected"]) > 1e-3
        or not item["a_log_fp32"]
        or not item["dt_bias_fp32"]
        for item in result.get("streaming", {}).get("checks", [])
    ):
        failures.append("streaming.correctness")
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))


if __name__ == "__main__":
    main()
