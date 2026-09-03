"""Inference-only pointwise fusion for native ComfyUI MiniMax-H3 blocks.

OpenVDN's tuned path fuses the large residual-stream RMSNorm + AdaLN affine and gated
residual stretches around attention and MLP. ComfyUI already optimizes the INT8 MLP
SwiGLU/down-projection path, so this module leaves the MLP itself native and only fuses
the remaining pointwise HBM passes.

The fused path is forward-only and resident-only. If compilation is unavailable or a
runtime shape is rejected, the state latches back to ComfyUI's native block forward;
there is intentionally no slower eager imitation of the fusion.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F


class _BlockFusionUnavailable(RuntimeError):
    pass


class BlockPointwiseCache:
    def __init__(self, limit: int = 16):
        self.limit = int(limit)
        self._indices: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()

    @staticmethod
    def _device_key(device: torch.device):
        return device.type, device.index

    @staticmethod
    def _fill_indices(segments, rows: int, device: torch.device):
        ids = torch.empty(rows, device=device, dtype=torch.long)
        covered = 0
        for raw_start, raw_stop, row in segments:
            start, stop = int(raw_start), int(raw_stop)
            if start != covered or stop < start or stop > rows:
                raise ValueError("MiniMax-H3 modulation segments must cover the sequence contiguously")
            count = stop - start
            if isinstance(row, torch.Tensor):
                if row.dtype != torch.long:
                    row = row.to(dtype=torch.long)
                if row.ndim == 0:
                    ids[start:stop].copy_(row.to(device=device).expand(count))
                elif row.numel() == count:
                    ids[start:stop].copy_(row.reshape(-1).to(device=device))
                else:
                    raise ValueError(
                        f"modulation row tensor for [{start}:{stop}] has {row.numel()} values, expected {count}"
                    )
            else:
                ids[start:stop].fill_(int(row))
            covered = stop
        if covered != rows:
            raise ValueError(f"MiniMax-H3 modulation segments cover {covered}/{rows} rows")
        return ids

    def indices(self, segments, rows: int, device: torch.device):
        scalar = all(not isinstance(row, torch.Tensor) for _, _, row in segments)
        if not scalar:
            return self._fill_indices(segments, rows, device)
        normalized = tuple((int(a), int(b), int(row)) for a, b, row in segments)
        key = (normalized, int(rows), self._device_key(device))
        cached = self._indices.get(key)
        if cached is not None:
            self._indices.move_to_end(key)
            return cached
        ids = self._fill_indices(normalized, rows, device)
        self._indices[key] = ids
        self._indices.move_to_end(key)
        while len(self._indices) > self.limit:
            self._indices.popitem(last=False)
        return ids

    def clear(self):
        self._indices.clear()

    release = clear


def _pre_body(hidden, weight, eps: float, scale, shift, indices):
    normed = F.rms_norm(hidden, (hidden.shape[-1],), weight, eps)
    selected_scale = scale.index_select(0, indices).to(normed.dtype)
    selected_shift = shift.index_select(0, indices).to(normed.dtype)
    return normed * (1.0 + selected_scale) + selected_shift


def _post_body(residual, gate, indices, branch_out):
    return residual + gate.index_select(0, indices).to(residual.dtype) * branch_out


def _compiled(state, name, fn, args, key):
    runtime = getattr(state, "branch_runtime", None)
    cache = getattr(runtime, "compiler", None)
    policy = getattr(state, "compile_policy", "off")
    if cache is None or policy == "off" or not args[0].is_cuda:
        raise _BlockFusionUnavailable("compiled pointwise runtime is unavailable")
    result = cache.call(name, fn, args, key, policy=policy)
    if result is None:
        raise _BlockFusionUnavailable(f"torch.compile rejected {name}")
    return result


def _weight_ready(norm, x: torch.Tensor) -> bool:
    weight = getattr(norm, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.device == x.device


def make_fast_block_forward(block: Any, state: Any, block_index: int):
    """Drop-in ComfyUI ``DiTBlock.forward`` with shared compiled pointwise kernels."""
    original_forward = block.forward

    def fallback(x, t_emb, mod_segments, rope_freqs, transformer_options):
        return original_forward(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options or {},
        )

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        if (
            not getattr(state, "block_fusion", False)
            or not getattr(state, "inference", False)
            or torch.is_grad_enabled()
            or not x.is_cuda
            or getattr(state, "weight_mode", "") != "resident"
            or not _weight_ready(block.norm1, x)
            or not _weight_ready(block.norm2, x)
        ):
            return fallback(x, t_emb, mod_segments, rope_freqs, transformer_options)

        original_x = x
        cache = getattr(state, "block_pointwise_cache", None)
        if cache is None:
            cache = BlockPointwiseCache()
            state.block_pointwise_cache = cache
        try:
            indices = cache.indices(mod_segments, x.shape[0], x.device)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
            base_key = (
                x.device.type,
                x.device.index,
                x.dtype,
                tuple(x.shape),
                tuple(indices.shape),
            )

            with state.diagnostics.scope("block.pre_attn", x.device):
                h = _compiled(
                    state,
                    "block_pre",
                    _pre_body,
                    (x, block.norm1.weight, float(block.norm1.eps), scale_msa, shift_msa, indices),
                    (*base_key, scale_msa.dtype),
                )
            attn_out = block.attn(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options or {},
            )
            with state.diagnostics.scope("block.post_attn", x.device):
                x = _compiled(
                    state,
                    "block_post",
                    _post_body,
                    (x, gate_msa, indices, attn_out),
                    (*base_key, gate_msa.dtype),
                )
            del h, attn_out

            with state.diagnostics.scope("block.pre_mlp", x.device):
                h = _compiled(
                    state,
                    "block_pre",
                    _pre_body,
                    (x, block.norm2.weight, float(block.norm2.eps), scale_mlp, shift_mlp, indices),
                    (*base_key, scale_mlp.dtype),
                )
            mlp_out = block.mlp(h)
            with state.diagnostics.scope("block.post_mlp", x.device):
                return _compiled(
                    state,
                    "block_post",
                    _post_body,
                    (x, gate_mlp, indices, mlp_out),
                    (*base_key, gate_mlp.dtype),
                )
        except _BlockFusionUnavailable as exc:
            state.block_fusion = False
            state.block_fusion_error = str(exc)
            return fallback(original_x, t_emb, mod_segments, rope_freqs, transformer_options)

    forward._vdn_h3_fast_block = True
    forward._vdn_h3_block_index = int(block_index)
    forward._vdn_h3_original = original_forward
    return forward


def install_block_fusions(model_patcher: Any, state: Any, blocks) -> int:
    if not getattr(state, "block_fusion", False):
        return 0
    existing = getattr(model_patcher, "object_patches", {})
    installed = 0
    for index, block in enumerate(blocks):
        path = f"diffusion_model.blocks.{index}.forward"
        if path in existing:
            raise RuntimeError(f"VDN-H3 fast block fusion collides with existing object patch {path}")
        model_patcher.add_object_patch(path, make_fast_block_forward(block, state, index))
        installed += 1
    return installed


def release_block_fusions(state: Any) -> None:
    cache = getattr(state, "block_pointwise_cache", None)
    if cache is not None:
        cache.release()


__all__ = [
    "BlockPointwiseCache",
    "install_block_fusions",
    "make_fast_block_forward",
    "release_block_fusions",
]
