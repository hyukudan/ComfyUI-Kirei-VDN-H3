"""Introspection helpers for repeatable VDN-H3 performance experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch


def _cuda_snapshot(model_patcher: Any) -> dict:
    raw = getattr(model_patcher, "load_device", "cpu")
    try:
        device = torch.device(raw)
    except (TypeError, RuntimeError):
        return {"device": str(raw), "available": False}
    result = {"device": str(device), "available": False}
    if device.type != "cuda" or not torch.cuda.is_available():
        return result
    try:
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            {
                "available": True,
                "name": props.name,
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_bytes": int(total),
                "free_bytes": int(free),
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _module_inventory(module: Any) -> dict:
    if module is None:
        return {"bytes": 0, "gib": 0.0, "dtypes": {}, "devices": {}}
    dtypes: dict[str, int] = {}
    devices: dict[str, int] = {}
    total = 0
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return {"bytes": 0, "gib": 0.0, "dtypes": {}, "devices": {}}
    for tensor in parameters():
        if tensor is None:
            continue
        size = int(tensor.numel() * tensor.element_size())
        total += size
        dtype = str(tensor.dtype).removeprefix("torch.")
        device = str(tensor.device)
        dtypes[dtype] = dtypes.get(dtype, 0) + size
        devices[device] = devices.get(device, 0) + size
    return {
        "bytes": total,
        "gib": total / 1024**3,
        "dtypes": dict(sorted(dtypes.items())),
        "devices": dict(sorted(devices.items())),
    }


def _projection_info(state):
    info = getattr(state, "projection_info", None)
    if info is None:
        return {"precision": getattr(state, "projection_precision", "bf16")}
    if is_dataclass(info):
        data = asdict(info)
    elif isinstance(info, dict):
        data = dict(info)
    else:
        data = {"value": str(info)}
    data["precision"] = getattr(state, "projection_precision", "bf16")
    original = data.get("original_bytes")
    quantized = data.get("quantized_bytes")
    if isinstance(original, int) and isinstance(quantized, int) and original:
        data["storage_ratio"] = quantized / original
        data["saved_bytes"] = original - quantized
    return data


def _stage_total(stages: dict, name: str) -> float:
    value = stages.get(name)
    if not isinstance(value, dict):
        return 0.0
    try:
        return float(value.get("total_ms", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _performance_analysis(state, diagnostics: dict, projection: dict, cuda: dict) -> dict:
    """Summarize the timing split without pretending overlapping stages are additive."""
    stages = diagnostics.get("stages", {}) if isinstance(diagnostics, dict) else {}
    softmax_ms = _stage_total(stages, "attention.softmax")
    linear_serial_ms = _stage_total(stages, "attention.linear_branch")
    linear_parallel_ms = _stage_total(stages, "attention.linear_branch.parallel")
    linear_ms = linear_parallel_ms or linear_serial_ms
    transfer_ms = _stage_total(stages, "weights.transfer")
    raw_copy_ms = _stage_total(stages, "parallel.raw_copy")
    join_ms = _stage_total(stages, "parallel.join")
    forward_ms = _stage_total(stages, "forward.total")

    # Sort only top-level scopes useful to a human profiler. Internal linear.* scopes
    # intentionally overlap their parent and therefore must not be summed together.
    ranked = []
    for name, item in stages.items():
        if not isinstance(item, dict) or name.startswith("linear."):
            continue
        try:
            ranked.append((name, float(item.get("total_ms", 0.0))))
        except (TypeError, ValueError):
            continue
    ranked.sort(key=lambda pair: pair[1], reverse=True)

    recommendations = []
    precision = projection.get("precision", "bf16")
    execution = getattr(state, "branch_execution", "serial")
    attention = getattr(state, "last_attention_backend", None)
    calibration_hit = getattr(getattr(state, "window_cache", None), "last_calibration_hit", None)
    if precision == "bf16" and linear_ms > 0 and linear_ms >= max(softmax_ms * 0.65, 1.0):
        recommendations.append(
            "The VDN linear branch is a large share of runtime while to_out_linear is BF16; "
            "benchmark the explicit workstation_fp8 profile on a large-VRAM GPU."
        )
    if attention == "grouped" and calibration_hit is None:
        recommendations.append(
            "Grouped attention is running without a calibration hit; benchmark grouped/Flex/FA2/FA4 "
            "for the exact render geometry before assuming grouped is fastest."
        )
    if (
        execution == "serial"
        and getattr(state, "weight_mode", "") == "resident"
        and cuda.get("available")
        and int(cuda.get("total_bytes", 0)) >= 48 * 1024**3
        and linear_ms > 0
    ):
        recommendations.append(
            "The branch is resident on a large-VRAM GPU but still serial; branch_execution=parallel can "
            "overlap exact VDN recurrence/projection with local softmax at the cost of extra raw Q/K/V VRAM."
        )
    if transfer_ms > max(linear_ms * 0.15, 2.0):
        recommendations.append(
            "Branch H2D transfer time is visible in the profile; prefer resident or hybrid placement if VRAM allows."
        )

    return {
        "forward_total_ms": forward_ms,
        "softmax_total_ms": softmax_ms,
        "linear_branch_total_ms": linear_ms,
        "weights_transfer_total_ms": transfer_ms,
        "parallel_raw_copy_total_ms": raw_copy_ms,
        "parallel_join_total_ms": join_ms,
        "linear_to_softmax_ratio": (linear_ms / softmax_ms) if softmax_ms > 0 else None,
        "top_level_stages": [
            {"name": name, "total_ms": elapsed} for name, elapsed in ranked[:10]
        ],
        "recommendations": recommendations,
        "note": (
            "CUDA stage totals on different streams may overlap. Compare them to locate bottlenecks; "
            "do not add parallel-stage totals to estimate wall time."
        ),
    }


def runtime_snapshot(model_patcher: Any) -> dict:
    """Return resolved runtime, fallback state and diagnostics without mutation."""
    state = getattr(model_patcher, "object_patches", {}).get(
        "diffusion_model._vdn_h3_state"
    )
    if state is None:
        raise RuntimeError("MODEL does not carry Kirei VDN-H3 state")
    broken = dict(getattr(getattr(state, "window_cache", None), "_broken", {}) or {})
    branch_bytes = int(state.weight_store.nbytes)
    calibration = getattr(getattr(state, "window_cache", None), "calibration", None)
    calibration_snapshot = calibration.snapshot() if calibration is not None else {}
    projection = _projection_info(state)
    diagnostics = state.diagnostics.snapshot(flush=True)
    cuda = _cuda_snapshot(model_patcher)
    return {
        "checkpoint": state.name,
        "forwards": int(state.forwards),
        "branch_mode": state.weight_mode,
        "branch_execution": getattr(state, "branch_execution", "serial"),
        "branch_bytes": branch_bytes,
        "branch_gib": branch_bytes / 1024**3,
        "branch_storage": state.weight_store.telemetry(),
        "attention_requested": state.attention_backend,
        "attention_last": state.last_attention_backend,
        "attention_failures": broken,
        "attention_calibration": {
            **calibration_snapshot,
            "last_hit": getattr(state.window_cache, "last_calibration_hit", None),
        },
        "kernel_backend": getattr(state, "kernel_backend", state.linear_kernels),
        "compile_policy": getattr(state, "compile_policy", "off"),
        "tile_frames": int(getattr(state, "tile_frames", 0)),
        "inference": bool(state.inference),
        "projection": projection,
        "lora_factors": _module_inventory(getattr(state, "lora_runtime", None)),
        "curve_factors": _module_inventory(getattr(state, "curve_adapter", None)),
        "diagnostics_enabled": bool(state.diagnostics.enabled),
        "diagnostics": diagnostics,
        "cuda": cuda,
        "performance_analysis": _performance_analysis(state, diagnostics, projection, cuda),
    }


__all__ = ["runtime_snapshot"]
