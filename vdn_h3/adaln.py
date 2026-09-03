"""Fidelity patch: AdaLN SiLU evaluated from the fp32 time embedding.

ComfyUI casts the fp32 time embedding to the compute dtype before every block's AdaLN
SiLU (``t_emb = time_embedder(t).to(dtype)``). OpenVDN patched diffusers to activate in
fp32 and cast afterwards, after measuring a 3.5e-3 norm-relative error on the AdaLN
projection that biases every block identically at every sampling step and therefore
accumulates along the trajectory.

Kirei keeps the fp32 embedding as it leaves the time embedder and lets every AdaLN
projection (the 50 DiT blocks and the final layer) activate that copy instead of the
rounded one. The projection GEMM itself still runs in the model's dtype, exactly like
the upstream patch. Curve-form (pruned) bases already run AdaLN in fp32 without a SiLU
and are left untouched.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


TIME_EMBEDDER_PATCH = "diffusion_model.time_embedder.forward"


def adaln_targets(dm: Any) -> list[str]:
    """Module paths (relative to the diffusion model) whose ``forward`` is replaced."""
    paths = [f"blocks.{index}.adaln_proj" for index in range(len(dm.blocks))]
    final = getattr(dm, "final_layer", None)
    if getattr(final, "adaln_proj", None) is not None:
        paths.append("final_layer.adaln_proj")
    return paths


def make_time_embedder_forward(embedder: Any, state: Any):
    original = embedder.forward

    def forward(t):
        out = original(t)
        state.adaln_source = out.detach() if isinstance(out, torch.Tensor) else None
        return out

    forward._vdn_h3_adaln_source = True
    forward._vdn_h3_original = original
    return forward


def make_adaln_forward(adaln: Any, state: Any):
    original = adaln.forward

    def forward(t_emb):
        if not getattr(adaln, "apply_silu", True):
            return original(t_emb)
        source = getattr(state, "adaln_source", None)
        if (
            isinstance(source, torch.Tensor)
            and source.shape == t_emb.shape
            and source.device == t_emb.device
        ):
            activated = F.silu(source.float())
        else:
            # No fp32 copy for this forward (a foreign caller): still activate in fp32
            # from the rounded input rather than in the compute dtype.
            activated = F.silu(t_emb.float())
        x = adaln.linear(activated.to(t_emb.dtype))
        x = x.view(x.shape[0] * adaln.modalities, adaln.expand * adaln.hidden)
        return x.chunk(adaln.expand, dim=-1)

    forward._vdn_h3_adaln_fp32 = True
    forward._vdn_h3_original = original
    return forward


def install_adaln_fp32(model_patcher: Any, state: Any, dm: Any) -> int:
    """Install the fp32 AdaLN patches; returns the number of patched projections."""
    if not getattr(state, "adaln_fp32", True):
        state.adaln_fp32 = False
        return 0
    embedder = getattr(dm, "time_embedder", None)
    if embedder is None or not hasattr(embedder, "forward"):
        state.adaln_fp32 = False
        return 0
    from .curve import is_curve_h3_base

    if is_curve_h3_base(dm):
        state.adaln_fp32 = False
        return 0
    targets = adaln_targets(dm)
    existing = getattr(model_patcher, "object_patches", {})
    paths = [TIME_EMBEDDER_PATCH] + [f"diffusion_model.{path}.forward" for path in targets]
    collisions = [path for path in paths if path in existing]
    if collisions:
        raise RuntimeError(
            f"VDN-H3 AdaLN fp32 patch collides with an existing object patch ({collisions[0]})"
        )
    model_patcher.add_object_patch(TIME_EMBEDDER_PATCH, make_time_embedder_forward(embedder, state))
    for path in targets:
        module = model_patcher.get_model_object(f"diffusion_model.{path}")
        model_patcher.add_object_patch(
            f"diffusion_model.{path}.forward", make_adaln_forward(module, state)
        )
    state.adaln_fp32 = True
    return len(targets)


__all__ = [
    "TIME_EMBEDDER_PATCH",
    "adaln_targets",
    "install_adaln_fp32",
    "make_adaln_forward",
    "make_time_embedder_forward",
]
