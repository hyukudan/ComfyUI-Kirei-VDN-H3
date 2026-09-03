"""Helpers for Comfy-managed VDN auxiliary models."""

from __future__ import annotations

from typing import Any

import torch


def _reject_multigpu_auxiliary(label: str):
    raise RuntimeError(
        "Kirei VDN-H3 currently supports one compute device per patched H3 model. "
        f"ComfyUI attempted to deep-clone the {label} auxiliary model for MultiGPU. "
        "Use the RTX PRO 6000 as the selected model device (or another single GPU) "
        "until VDN's distributed/Ulysses branch is ported; sharing these runtime "
        "closures across independent model clones would be incorrect."
    )


def create_auxiliary_patcher(
    module: Any,
    base_patcher: Any,
    *,
    size: int,
    label: str,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
):
    """Wrap storage in ModelPatcher and fail clearly on unsupported deep MultiGPU clones."""
    from comfy.model_patcher import ModelPatcher

    if load_device is None:
        load_device = getattr(base_patcher, "load_device", torch.device("cpu"))
    if offload_device is None:
        offload_device = getattr(base_patcher, "offload_device", torch.device("cpu"))
    patcher = ModelPatcher(
        module,
        load_device=torch.device(load_device),
        offload_device=torch.device(offload_device),
        size=int(size),
    )
    # ModelPatcher.deepclone_multigpu requires cached_patcher_init. Supplying an
    # intentional rejecting factory turns an otherwise obscure factory error (or,
    # worse, a shallow closure bound to the wrong model) into a deterministic guard.
    patcher.cached_patcher_init = (_reject_multigpu_auxiliary, (str(label),))
    return patcher


__all__ = ["create_auxiliary_patcher"]
