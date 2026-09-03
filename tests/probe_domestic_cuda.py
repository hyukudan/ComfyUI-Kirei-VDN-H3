"""Synthetic CUDA probe for consumer/workstation VDN-H3 paths.

Run inside the ComfyUI Python environment, for example:

    python tests/probe_domestic_cuda.py --device cuda:0 --json vdn-domestic.json

No model weights are needed. The probe checks the exact tiled branch, hybrid transfer
policy and the quantized VDN projection families on the selected GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import time
from pathlib import Path

import torch

from vdn_h3 import window
from vdn_h3.projection import (
    FP8_SCALE_KEY,
    FP8_WEIGHT_KEY,
    fp8_supported,
    int8_supported,
    prepare_projection_maps,
    project,
)
from vdn_h3.runtime import OptimizedLinearBranch, SharedBranchRuntime
from vdn_h3.weights import ManagedBranchWeights


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
    flat_got = got.float().reshape(-1)
    flat_want = want.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_got, flat_want, dim=0)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": float(cosine.item()),
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

    untiled_runtime = SharedBranchRuntime(kernel_backend="auto", compile_policy="off", tile_frames=0)
    tiled_runtime = SharedBranchRuntime(kernel_backend="auto", compile_policy="off", tile_frames=5)
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
    store = ManagedBranchWeights(blocks, mode="hybrid", pin_strategy="auto")
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


def _info_dict(info):
    if is_dataclass(info):
        raw = asdict(info)
    elif hasattr(info, "__dict__"):
        raw = dict(info.__dict__)
    else:
        raw = {"value": str(info)}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(value, (str, int, float, bool))
    }


def _probe_quantized_projection(device, precision, quick=False):
    torch.manual_seed(1203 if precision == "fp8" else 1204)
    available = fp8_supported(device, probe=True) if precision == "fp8" else int8_supported(device, probe=True)
    if not available:
        return {"available": False, "precision": precision}

    # Keep K divisible by the 256 ConvRot group on the INT8 path.
    rows, in_features, out_features = (128, 256, 384) if quick else (512, 1024, 1536)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.03
    maps, info = prepare_projection_maps(
        [{"to_out_linear.weight": weight}],
        device,
        precision,
        skip_end_blocks=0,
    )
    quantized = maps[0]
    x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
    bf16_weight = weight.to(device)
    def run_reference():
        with torch.inference_mode():
            return torch.nn.functional.linear(x, bf16_weight)

    reference, bf16_ms = _time(
        device, run_reference, runs=4 if quick else 10
    )
    gpu_weights = {
        FP8_WEIGHT_KEY: quantized[FP8_WEIGHT_KEY].to(device),
        FP8_SCALE_KEY: quantized[FP8_SCALE_KEY].to(device),
    }
    def run_quantized():
        with torch.inference_mode():
            return project(x, gpu_weights)

    got, quant_ms = _time(device, run_quantized, runs=4 if quick else 10)
    error = _error(
        got,
        reference,
        atol=8e-2 if precision == "fp8" else 1.5e-1,
        rtol=1e-1 if precision == "fp8" else 1.5e-1,
    )
    return {
        "available": True,
        "precision": precision,
        **error,
        "bf16_ms": bf16_ms,
        "quantized_ms": quant_ms,
        "speedup_vs_bf16": bf16_ms / quant_ms if quant_ms > 0 else None,
        "original_bytes": info.original_bytes,
        "quantized_bytes": info.quantized_bytes,
        "storage_ratio": info.quantized_bytes / info.original_bytes if info.original_bytes else None,
        "info": _info_dict(info),
    }


def probe_fp8(device, quick=False):
    return _probe_quantized_projection(device, "fp8", quick)


def probe_int8(device, quick=False):
    return _probe_quantized_projection(device, "int8", quick)


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
        ("int8_convrot_projection", lambda: probe_int8(device, args.quick)),
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
    for name in ("int8_convrot_projection", "fp8_projection"):
        item = result.get(name, {})
        if not item.get("available"):
            continue
        # Quantized inference changes arithmetic. Require strong local directional
        # agreement rather than exact BF16 elementwise equality.
        if float(item.get("cosine", 0.0)) < 0.98:
            failures.append(name)
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))


if __name__ == "__main__":
    main()
