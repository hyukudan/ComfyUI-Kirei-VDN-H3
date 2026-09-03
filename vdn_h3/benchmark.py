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
    return {
        "checkpoint": state.name,
        "forwards": int(state.forwards),
        "branch_mode": state.weight_mode,
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
        "projection": _projection_info(state),
        "lora_factors": _module_inventory(getattr(state, "lora_runtime", None)),
        "curve_factors": _module_inventory(getattr(state, "curve_adapter", None)),
        "diagnostics_enabled": bool(state.diagnostics.enabled),
        "diagnostics": state.diagnostics.snapshot(flush=True),
        "cuda": _cuda_snapshot(model_patcher),
    }


__all__ = ["runtime_snapshot"]
