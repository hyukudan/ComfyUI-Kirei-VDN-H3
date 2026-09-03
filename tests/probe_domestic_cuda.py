"""Synthetic CUDA probe for consumer/workstation VDN-H3 paths.

Run inside the ComfyUI Python environment, for example:

    python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic.json

No model weights are needed. The probe checks the exact tiled branch, hybrid transfer
policy and opt-in FP8 projection on the selected GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vdn_h3 import window
from vdn_h3.fp8 import (
    FP8_SCALE_KEY,
    FP8_WEIGHT_KEY,
    fp8_supported,
    prepare_projection_maps,
    project,
)
from vdn_h3.runtime import OptimizedLinearBranch, SharedBranchRuntime
from vdn_h3.weights import FP8_STREAMED_PROJECTION_KEY, ManagedBranchWeights


def _sync(device):
    torch.cuda.synchronize(device)


def _time(device, fn, warmup=1, runs=4):
    for _ in range(warmup):
        value = fn()
        del value
    _sync(device)
    start = time.perf_counter()
    value = None
    for _ in range(runs):
        value = fn()
    _sync(device)
    return value, (time.perf_counter() - start) * 1000.0 / runs


def _error(got, want, atol=3e-2, rtol=5e-2):
    diff = (got.float() - want.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose": bool(torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol)),
    }


def _branch_weights(device, hidden=256, heads=4, dim=32, short=True):
    channels = heads * dim
    cpu = {
        "to_out_linear.weight": torch.randn(hidden, channels, dtype=torch.bfloat16) * 0.05,
        "beta_proj.weight": torch.randn(heads, hidden, dtype=torch.bfloat16) * 0.05,
        "alpha.down.weight": torch.randn(dim, hidden, dtype=torch.bfloat16) * 0.05,
        "alpha.up.weight": torch.randn(channels, dim, dtype=torch.bfloat16) * 0.05,
        "alpha.dt_bias": torch.randn(channels, dtype=torch.float32) * 0.05,
        "alpha.A_log": torch.zeros(heads, dtype=torch.float32),
        "output_gate.down.weight": torch.randn(dim, hidden, dtype=torch.bfloat16) * 0.05,
        "output_gate.up.weight": torch.randn(channels, dim, dtype=torch.bfloat16) * 0.05,
        "output_gate.up.bias": torch.randn(channels, dtype=torch.bfloat16) * 0.05,
        "norm.weight": torch.ones(dim, dtype=torch.bfloat16),
    }
    if short:
        cpu.update(
            {
                "short_conv.k_sp.weight": torch.randn(channels, 1, 3, 3, dtype=torch.bfloat16) * 0.02,
                "short_conv.k_tm.weight": torch.randn(channels, 1, 5, dtype=torch.bfloat16) * 0.02,
                "short_conv.v_sp.weight": torch.randn(channels, 1, 3, 3, dtype=torch.bfloat16) * 0.02,
                "short_conv.v_tm.weight": torch.randn(channels, 1, 5, dtype=torch.bfloat16) * 0.02,
            }
        )
    return {key: value.to(device) for key, value in cpu.items()}


def probe_tiled(device, quick=False):
    torch.manual_seed(1201)
    frames = 7 if quick else 17
    height = width = 4 if quick else 8
    per_frame = height * width
    heads, dim, hidden = 4, 32, 256
    rows = frames * per_frame
    weights = _branch_weights(device, hidden, heads, dim, short=True)
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    q = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bounds = window.window_bounds(frames, 1, 5)

    untiled_runtime = SharedBranchRuntime(
        kernel_backend="auto", compile_policy="off", tile_frames=0
    )
    tiled_runtime = SharedBranchRuntime(
        kernel_backend="auto", compile_policy="off", tile_frames=5
    )
    untiled = OptimizedLinearBranch(
        weights, heads, dim, short_conv=("k", "v"), enable_text_state=False,
        kernel_backend="auto", compile_policy="off", tile_frames=0,
    ).set_runtime(runtime_cache=untiled_runtime)
    tiled = OptimizedLinearBranch(
        weights, heads, dim, short_conv=("k", "v"), enable_text_state=False,
        kernel_backend="auto", compile_policy="off", tile_frames=5,
    ).set_runtime(runtime_cache=tiled_runtime)

    def run(module):
        with torch.no_grad():
            return module.projected_delta(
                weights, x, q, k, v, frames, per_frame, bounds,
                frame_size=(height, width), skip_ends=True, inference=True,
            )

    reference, reference_ms = _time(device, lambda: run(untiled), runs=2 if quick else 4)
    got, tiled_ms = _time(device, lambda: run(tiled), runs=2 if quick else 4)
    return {
        "shape": list(got.shape),
        **_error(got, reference, atol=3e-2, rtol=5e-2),
        "untiled_ms": reference_ms,
        "tiled_ms": tiled_ms,
    }


def probe_hybrid_stream(device, quick=False):
    torch.manual_seed(1202)
    mib = 8 if quick else 64
    elements = mib * 1024 * 1024 // 2
    blocks = [
        {
            "to_out_linear.weight": torch.full((elements,), float(index + 1), dtype=torch.bfloat16),
            "small": torch.full((1024,), float(index), dtype=torch.bfloat16),
        }
        for index in range(4)
    ]
    store = ManagedBranchWeights(
        blocks, mode="hybrid", pin_strategy="auto"
    )
    checks = []
    for index in range(4):
        store.prefetch(index, device, torch.bfloat16, None)
        weights = store.weights_on(index, device, torch.bfloat16, None)
        observed = float(weights["to_out_linear.weight"][:1024].float().mean().item())
        checks.append({"block": index, "observed": observed, "expected": float(index + 1)})
        store.mark_consumed(index, device, torch.bfloat16)
    _sync(device)
    telemetry = store.telemetry()
    store.release()
    return {
        "source_mib_per_projection": mib,
        "checks": checks,
        "telemetry": telemetry,
        "correct": all(abs(x["observed"] - x["expected"]) < 1e-3 for x in checks),
    }


def probe_fp8(device, quick=False):
    torch.manual_seed(1203)
    if not fp8_supported(device, probe=True):
        return {"available": False}
    rows, in_features, out_features = (128, 256, 384) if quick else (512, 1024, 1536)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.03
    maps, info = prepare_projection_maps([{"to_out_linear.weight": weight}], device)
    quantized = maps[0]
    x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
    bf16_weight = weight.to(device)
    reference, bf16_ms = _time(
        device, lambda: torch.nn.functional.linear(x, bf16_weight), runs=4 if quick else 10
    )
    gpu_weights = {
        FP8_WEIGHT_KEY: quantized[FP8_WEIGHT_KEY].to(device),
        FP8_SCALE_KEY: quantized[FP8_SCALE_KEY].to(device),
    }
    got, fp8_ms = _time(device, lambda: project(x, gpu_weights), runs=4 if quick else 10)
    return {
        "available": True,
        **_error(got, reference, atol=8e-2, rtol=1e-1),
        "bf16_ms": bf16_ms,
        "fp8_ms": fp8_ms,
        "original_bytes": info.original_bytes,
        "quantized_bytes": info.quantized_bytes,
        "per_tensor": info.per_tensor,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    result = {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "quick": bool(args.quick),
    }
    for name, fn in (
        ("tiled", lambda: probe_tiled(device, args.quick)),
        ("hybrid_stream", lambda: probe_hybrid_stream(device, args.quick)),
        ("fp8_projection", lambda: probe_fp8(device, args.quick)),
    ):
        try:
            result[name] = fn()
        except Exception as exc:
            result[name] = {"fatal_error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        path = Path(args.json_path)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")
    failures = []
    if not result.get("tiled", {}).get("allclose", False):
        failures.append("tiled")
    if not result.get("hybrid_stream", {}).get("correct", False):
        failures.append("hybrid_stream")
    fp8 = result.get("fp8_projection", {})
    if fp8.get("available") and not fp8.get("allclose", False):
        failures.append("fp8_projection")
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))


if __name__ == "__main__":
    main()
