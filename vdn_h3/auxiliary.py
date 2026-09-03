"""Helpers for Comfy-managed VDN auxiliary models."""

from __future__ import annotations

from typing import Any

import torch


def _reject_multigpu_auxiliary(label: str):
    raise RuntimeError(
        "Kirei VDN-H3 currently supports one compute device per patched H3 model. "
        f"ComfyUI attempted to deep-clone the {label} auxiliary model for MultiGPU. "
        "Select one GPU for the patched H3 model until VDN's distributed/Ulysses "
        "execution is implemented; sharing runtime closures across independent model "
        "clones would be incorrect."
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
    # intentional rejecting factory turns an obscure clone error (or a shallow closure
    # bound to the wrong model) into a deterministic guard.
    patcher.cached_patcher_init = (_reject_multigpu_auxiliary, (str(label),))
    return patcher


def unload_auxiliary(patcher: Any, module: Any) -> bool:
    """Unload an auxiliary through ComfyUI first, with a CPU fallback.

    Direct ``module.to('cpu')`` can leave ComfyUI's loaded-model bookkeeping stale.
    This helper keeps the lifecycle coherent whenever the active ComfyUI exposes the
    normal unload API, while remaining usable in dependency-light tests.
    """
    if patcher is not None:
        try:
            import comfy.model_management as model_management

            model_management.unload_model_and_clones(
                patcher, unload_additional_models=False
            )
            return True
        except Exception:
            pass
    try:
        module.to(device="cpu")
        return True
    except Exception:
        return False


__all__ = ["create_auxiliary_patcher", "unload_auxiliary"]
