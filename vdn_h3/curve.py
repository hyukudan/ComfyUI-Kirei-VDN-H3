"""Runtime low-rank injection for pruned MiniMax-H3 AdaLN curves."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .auxiliary import create_auxiliary_patcher, unload_auxiliary


_CURVE_CACHE: ContextVar[dict | None] = ContextVar("kirei_vdn_curve_cache", default=None)
_EGRID_NAME = "h3_silu_temb_grid.safetensors"


@contextmanager
def curve_runtime_scope():
    token = _CURVE_CACHE.set({})
    try:
        yield
    finally:
        _CURVE_CACHE.reset(token)


def _egrid_candidates(path: str | None, checkpoint_root: str | None):
    seen = set()

    def emit(value):
        if value is None:
            return
        resolved = Path(value).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved

    if path is not None:
        yield from emit(path)
        return
    if checkpoint_root is not None:
        root = Path(checkpoint_root).resolve()
        yield from emit(root / _EGRID_NAME)
        yield from emit(root / "linear_branch" / _EGRID_NAME)
        yield from emit(root.parent / _EGRID_NAME)
    try:
        import folder_paths

        for root in folder_paths.get_folder_paths("vdn"):
            yield from emit(Path(root) / _EGRID_NAME)
    except Exception:
        pass
    # Legacy compatibility: older installations obtained the grid from this sibling
    # node. It is intentionally last; a self-contained VDN checkpoint takes priority.
    custom_nodes = Path(__file__).resolve().parents[2]
    yield from emit(custom_nodes / "ComfyUI-MiniMax-H3-Turbo" / _EGRID_NAME)


def load_egrid(path: str | None = None, *, checkpoint_root: str | None = None) -> torch.Tensor:
    candidates = list(_egrid_candidates(path, checkpoint_root))
    source = next(
        (candidate for candidate in candidates if candidate.is_file() and candidate.suffix.lower() == ".safetensors"),
        None,
    )
    if source is None:
        searched = ", ".join(str(candidate) for candidate in candidates) or _EGRID_NAME
        raise FileNotFoundError(
            "pruned H3 AdaLN adapters require h3_silu_temb_grid.safetensors. "
            f"Searched: {searched}"
        )
    from safetensors.torch import load_file

    state = load_file(str(source), device="cpu")
    if set(state) != {"silu_t_emb_grid"}:
        raise ValueError(f"unexpected e-grid inventory in {source}: {sorted(state)}")
    grid = state["silu_t_emb_grid"].detach().to(torch.float32).contiguous()
    if grid.ndim != 2 or grid.shape[0] < 2:
        raise ValueError(f"invalid silu_t_emb_grid shape {tuple(grid.shape)}")
    return grid


def _compute_dtype(model_patcher):
    try:
        dtype = model_patcher.model_dtype()
    except Exception:
        dtype = None
    return dtype if dtype in {torch.float16, torch.bfloat16} else torch.bfloat16


def is_curve_h3_base(dm) -> bool:
    """Detect pruned/curve H3 bases by flag or collapsed AdaLN input width.

    Some converted/pruned checkpoints do not reliably expose ``use_adaln_curves`` even
    though ``adaln_proj.linear`` has already been collapsed onto the small shared curve
    coordinate.  The structural weight shape is therefore used as a second, independent
    signal.  Unlike ports that simply skip those LoRA terms, Kirei keeps the exact
    e-grid reinjection path and only uses this helper to select it reliably.
    """
    if bool(getattr(dm, "use_adaln_curves", False)):
        return True
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        blocks = getattr(dm, "transformer_blocks", None)
    try:
        block = blocks[0]
        weight = block.adaln_proj.linear.weight
        return bool(weight.ndim == 2 and int(weight.shape[-1]) < 64)
    except (AttributeError, IndexError, TypeError):
        return False


class CurveAdapterState(nn.Module):
    """Per-model low-rank AdaLN terms exposed as an auxiliary Comfy model."""

    def __init__(
        self,
        egrid: torch.Tensor,
        terms: Iterable[tuple[Any, float]],
        *,
        storage_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = torch.device("cpu")
        self.register_parameter("egrid", nn.Parameter(egrid.to(torch.float32), requires_grad=False))
        self._terms: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        self._patcher = None
        for index, (patch, strength) in enumerate(terms):
            dtype = storage_dtype or torch.promote_types(patch.up.dtype, patch.down.dtype)
            up_name, down_name = f"up_{index}", f"down_{index}"
            self.register_parameter(
                up_name,
                nn.Parameter(
                    patch.up.detach().to("cpu", dtype=dtype).contiguous(), requires_grad=False
                ),
            )
            self.register_parameter(
                down_name,
                nn.Parameter(
                    patch.down.detach().to("cpu", dtype=dtype).contiguous(), requires_grad=False
                ),
            )
            self._terms[patch.key].append(
                (up_name, down_name, float(strength) * float(patch.scale))
            )

    @property
    def targets(self):
        return tuple(sorted(self._terms))

    @property
    def nbytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def release(self):
        unload_auxiliary(self._patcher, self)
        return self

    def full_embedding(self, t_emb, table):
        cache = _CURVE_CACHE.get()
        key = (t_emb.data_ptr(), tuple(t_emb.shape), t_emb.device.type, t_emb.device.index)
        if cache is not None and key in cache:
            return cache[key]
        coords = t_emb.detach().to(torch.float32)
        table = table.to(device=t_emb.device, dtype=torch.float32)
        grid = self.egrid.to(device=t_emb.device, dtype=torch.float32)
        if table.ndim != 2 or coords.shape[1] != table.shape[1]:
            raise RuntimeError(
                f"AdaLN curve coordinate shape {tuple(coords.shape)} is incompatible with table {tuple(table.shape)}"
            )
        if grid.shape[0] != table.shape[0]:
            raise RuntimeError(
                f"e-grid rows {grid.shape[0]} do not match H3 curve table rows {table.shape[0]}"
            )
        nearest = torch.cdist(coords, table).argmin(dim=1)

        def candidate(lo, hi):
            start, end = table[lo], table[hi]
            direction = end - start
            fraction = ((coords - start) * direction).sum(1) / direction.square().sum(1).clamp_min(1e-20)
            fraction = fraction.clamp(0.0, 1.0)
            reconstruction = torch.lerp(start, end, fraction[:, None])
            error = (reconstruction - coords).square().sum(1)
            return fraction, error

        left_lo = (nearest - 1).clamp_min(0)
        left_hi = nearest
        right_lo = nearest
        right_hi = (nearest + 1).clamp_max(table.shape[0] - 1)
        left_fraction, left_error = candidate(left_lo, left_hi)
        right_fraction, right_error = candidate(right_lo, right_hi)
        choose_left = left_error <= right_error
        lo = torch.where(choose_left, left_lo, right_lo)
        hi = torch.where(choose_left, left_hi, right_hi)
        fraction = torch.where(choose_left, left_fraction, right_fraction)
        full = torch.lerp(grid[lo], grid[hi], fraction[:, None])
        if cache is not None:
            cache[key] = full
        return full

    def delta(self, key, full_embedding, dtype):
        terms = self._terms.get(key)
        if not terms:
            raise KeyError(f"no curve AdaLN terms registered for {key!r}")
        result = None
        input_value = full_embedding.to(dtype)
        for up_name, down_name, scale in terms:
            up = getattr(self, up_name)
            down = getattr(self, down_name)
            if up.device != full_embedding.device or up.dtype != dtype:
                up = up.to(full_embedding.device, dtype)
            if down.device != full_embedding.device or down.dtype != dtype:
                down = down.to(full_embedding.device, dtype)
            value = F.linear(F.linear(input_value, down), up) * scale
            result = value if result is None else result.add(value)
        return result

    def forward(self, *args, **kwargs):
        raise RuntimeError("CurveAdapterState is storage-only")


def make_curve_adaln_forward(base, dm, state: CurveAdapterState, key: str):
    original = base.forward

    def forward(t_emb):
        outputs = original(t_emb)
        full = state.full_embedding(t_emb, dm.adaln_t_table)
        delta = state.delta(key, full, outputs[0].dtype)
        delta = delta.view(t_emb.shape[0] * base.modalities, base.expand * base.hidden)
        chunks = delta.chunk(base.expand, dim=-1)
        return tuple(output + addition for output, addition in zip(outputs, chunks))

    forward._kirei_vdn_curve_adaln = True
    return forward


def apply_curve_adapters(model_patcher, vdn_state, terms, *, egrid_path: str | None = None):
    terms = list(terms)
    if not terms:
        return 0
    dm = model_patcher.get_model_object("diffusion_model")
    if not is_curve_h3_base(dm):
        raise RuntimeError("curve AdaLN factors were produced for a non-curve H3 base")
    if getattr(dm, "adaln_t_table", None) is None:
        raise RuntimeError(
            "pruned H3 AdaLN structure was detected but the shared adaln_t_table is missing"
        )
    curve_state = CurveAdapterState(
        load_egrid(egrid_path, checkpoint_root=getattr(vdn_state, "checkpoint_root", None)),
        terms,
        storage_dtype=_compute_dtype(model_patcher),
    )
    if curve_state.egrid.shape[1] != terms[0][0].down.shape[1]:
        raise RuntimeError(
            f"e-grid width {curve_state.egrid.shape[1]} does not match adapter input width {terms[0][0].down.shape[1]}"
        )
    vdn_state.curve_adapter = curve_state
    try:
        patcher = create_auxiliary_patcher(
            curve_state,
            model_patcher,
            size=curve_state.nbytes,
            label="VDN curve adapter factors",
        )
        curve_state._patcher = patcher
        setter = getattr(model_patcher, "set_additional_models", None)
        if setter is not None:
            setter("vdn_h3_curve", [patcher])
    except ImportError:
        pass

    existing = getattr(model_patcher, "object_patches", {})
    for key in curve_state.targets:
        module_path = key.removesuffix(".linear.weight")
        forward_path = module_path + ".forward"
        if forward_path in existing:
            raise RuntimeError(f"curve AdaLN patch collides with {forward_path}")
        base = model_patcher.get_model_object(module_path)
        model_patcher.add_object_patch(
            forward_path, make_curve_adaln_forward(base, dm, curve_state, key)
        )
    return len(curve_state.targets)


__all__ = [
    "CurveAdapterState",
    "apply_curve_adapters",
    "curve_runtime_scope",
    "is_curve_h3_base",
    "load_egrid",
]
