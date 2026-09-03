"""Fused pointwise kernels on the local-softmax side of VDN-H3."""

from __future__ import annotations

import torch


def _gate_flatten_body(softmax_out, gate):
    return (softmax_out * gate.to(softmax_out.dtype)).reshape(softmax_out.shape[0], -1)


def apply_softmax_gate(softmax_out, gate, state, *, inference: bool):
    """Apply the per-head gate and store directly in the flattened out-proj layout."""
    if not inference or not softmax_out.is_cuda:
        return _gate_flatten_body(softmax_out, gate)
    runtime = getattr(state, "branch_runtime", None)
    cache = getattr(runtime, "compiler", None)
    policy = getattr(state, "compile_policy", "off")
    if cache is not None and policy != "off":
        key = (
            softmax_out.device.type,
            softmax_out.device.index,
            softmax_out.dtype,
            tuple(softmax_out.shape),
            gate.dtype,
            tuple(gate.shape),
        )
        result = cache.call(
            "softmax_gate_flatten",
            _gate_flatten_body,
            (softmax_out, gate),
            key,
            policy=policy,
        )
        if result is not None:
            return result
    return _gate_flatten_body(softmax_out, gate)


__all__ = ["apply_softmax_gate"]
