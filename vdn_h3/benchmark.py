"""Introspection helpers for repeatable VDN-H3 performance experiments."""

from __future__ import annotations

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


def runtime_snapshot(model_patcher: Any) -> dict:
    """Return resolved runtime, backend fallbacks and diagnostics without mutation."""
    state = getattr(model_patcher, "object_patches", {}).get(
        "diffusion_model._vdn_h3_state"
    )
    if state is None:
        raise RuntimeError("MODEL does not carry Kirei VDN-H3 state")
    broken = dict(getattr(getattr(state, "window_cache", None), "_broken", {}) or {})
    branch_bytes = int(state.weight_store.nbytes)
    return {
        "checkpoint": state.name,
        "forwards": int(state.forwards),
        "branch_mode": state.weight_mode,
        "branch_bytes": branch_bytes,
        "branch_gib": branch_bytes / 1024**3,
        "attention_requested": state.attention_backend,
        "attention_last": state.last_attention_backend,
        "attention_failures": broken,
        "linear_kernels": state.linear_kernels,
        "inference": bool(state.inference),
        "lora_factors": _module_inventory(getattr(state, "lora_runtime", None)),
        "curve_factors": _module_inventory(getattr(state, "curve_adapter", None)),
        "diagnostics_enabled": bool(state.diagnostics.enabled),
        "diagnostics": state.diagnostics.snapshot(),
        "cuda": _cuda_snapshot(model_patcher),
    }


__all__ = ["runtime_snapshot"]
