"""Introspection helpers for repeatable VDN-H3 performance experiments."""

from __future__ import annotations

from typing import Any


def runtime_snapshot(model_patcher: Any) -> dict:
    """Return the optimized node's resolved runtime and diagnostics without mutation."""
    state = getattr(model_patcher, "object_patches", {}).get(
        "diffusion_model._vdn_h3_private_state"
    )
    if state is None:
        raise RuntimeError("MODEL does not carry Kirei VDN-H3 state")
    return {
        "checkpoint": state.name,
        "forwards": state.forwards,
        "branch_mode": state.weight_mode,
        "branch_bytes": state.weight_store.nbytes,
        "attention_requested": state.attention_backend,
        "attention_last": state.last_attention_backend,
        "linear_kernels": state.linear_kernels,
        "inference": state.inference,
        "diagnostics": state.diagnostics.snapshot(),
    }


__all__ = ["runtime_snapshot"]
