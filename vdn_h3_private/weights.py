"""ComfyUI-accounted VDN branch-weight storage and asynchronous streaming."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


FP32_BRANCH_KEYS = frozenset({"alpha.A_log", "alpha.dt_bias"})


class _WeightBlock(nn.Module):
    def __init__(self, weights: Mapping[str, torch.Tensor], block_index: int):
        super().__init__()
        self._names: dict[str, str] = {}
        for tensor_index, (key, tensor) in enumerate(sorted(weights.items())):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"branch block {block_index} tensor {key!r} is {type(tensor).__name__}, not Tensor"
                )
            name = f"w{tensor_index}"
            self.register_parameter(
                name,
                nn.Parameter(tensor.detach().to(device="cpu").contiguous(), requires_grad=False),
            )
            self._names[str(key)] = name

    def keys(self):
        return self._names.keys()

    def tensor(self, key: str) -> torch.Tensor:
        return getattr(self, self._names[key])


class BranchWeightsModel(nn.Module):
    """A real nn.Module so ComfyUI can account, clone, load and offload VDN weights."""

    def __init__(self, weights: Sequence[Mapping[str, torch.Tensor]]):
        super().__init__()
        if not weights:
            raise ValueError("VDN branch weights cannot be empty")
        self.blocks = nn.ModuleList([_WeightBlock(block, index) for index, block in enumerate(weights)])
        self.device = torch.device("cpu")

    def block(self, index: int) -> _WeightBlock:
        return self.blocks[index]

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("BranchWeightsModel is storage-only and has no forward")


@dataclass
class _StreamSlot:
    tensors: dict[str, torch.Tensor]
    block: int | None = None
    ready: Any = None
    consumed: Any = None


class _StreamContext:
    def __init__(self, device: torch.device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = [_StreamSlot({}), _StreamSlot({})]
        self.block_to_slot: dict[int, int] = {}


class ManagedBranchWeights(nn.Module):
    """Own branch weights with either Comfy-managed residency or double-buffered stream."""

    def __init__(self, weights: Sequence[Mapping[str, torch.Tensor]], mode: str = "stream"):
        super().__init__()
        if mode not in {"stream", "resident"}:
            raise ValueError(f"branch weight mode must be 'stream' or 'resident', got {mode!r}")
        self.mode = mode
        self.model = BranchWeightsModel(weights)
        self._closed = False
        self._contexts: dict[tuple[str, int | None, torch.dtype], _StreamContext] = {}
        self._patcher = None
        if mode == "stream":
            self._pin_cpu_weights()

    def __len__(self) -> int:
        return len(self.model.blocks)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def _pin_cpu_weights(self) -> None:
        try:
            import comfy.model_management as model_management
        except ImportError:
            model_management = None
        for param in self.model.parameters():
            if param.device.type != "cpu" or param.is_pinned():
                continue
            pinned = False
            if model_management is not None:
                try:
                    pinned = bool(model_management.pin_memory(param.data))
                except Exception:
                    pinned = False
            if not pinned and torch.cuda.is_available():
                try:
                    param.data = param.data.pin_memory()
                except Exception:
                    pass

    def attach_to(self, base_patcher: Any, key: str = "vdn_h3_branch"):
        try:
            from comfy.model_patcher import ModelPatcher
        except ImportError:
            return None
        if self._patcher is None:
            if self.mode == "resident":
                load_device = getattr(base_patcher, "load_device", torch.device("cpu"))
                offload_device = getattr(base_patcher, "offload_device", torch.device("cpu"))
            else:
                load_device = offload_device = torch.device("cpu")
            self._patcher = ModelPatcher(
                self.model,
                load_device=load_device,
                offload_device=offload_device,
                size=self.nbytes,
            )
        setter = getattr(base_patcher, "set_additional_models", None)
        if setter is not None:
            setter(key, [self._patcher])
        return self._patcher

    @staticmethod
    def _target_dtype(key: str, compute_dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if key in FP32_BRANCH_KEYS else compute_dtype

    def _source(self, block_index: int, keys: Collection[str] | None) -> dict[str, torch.Tensor]:
        if not 0 <= block_index < len(self):
            raise IndexError(f"VDN-H3 branch block {block_index} is out of range")
        block = self.model.block(block_index)
        wanted = tuple(block.keys()) if keys is None else tuple(sorted(set(keys)))
        missing = sorted(set(wanted) - set(block.keys()))
        if missing:
            raise KeyError(f"VDN-H3 branch block {block_index} is missing requested weights {missing}")
        return {key: block.tensor(key) for key in wanted}

    def _context(self, device: torch.device, dtype: torch.dtype) -> _StreamContext:
        key = (device.type, device.index, dtype)
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = _StreamContext(device)
            self._contexts[key] = ctx
        return ctx

    def _schedule(self, ctx, slot_index, block_index, compute_dtype, keys):
        slot = ctx.slots[slot_index]
        if slot.block == block_index and (keys is None or set(keys).issubset(slot.tensors)):
            return
        source = self._source(block_index, keys)
        if slot.consumed is not None:
            ctx.stream.wait_event(slot.consumed)
        with torch.cuda.stream(ctx.stream):
            for key, tensor in source.items():
                target_dtype = self._target_dtype(key, compute_dtype)
                target = slot.tensors.get(key)
                if (
                    target is None
                    or target.shape != tensor.shape
                    or target.dtype != target_dtype
                    or target.device != ctx.device
                ):
                    target = torch.empty(tensor.shape, device=ctx.device, dtype=target_dtype)
                    slot.tensors[key] = target
                target.copy_(tensor, non_blocking=tensor.is_pinned())
            ready = torch.cuda.Event()
            ready.record(ctx.stream)
        if slot.block is not None:
            ctx.block_to_slot.pop(slot.block, None)
        slot.block = block_index
        slot.ready = ready
        slot.consumed = None
        ctx.block_to_slot[block_index] = slot_index

    def _stream_weights(self, block_index, device, dtype, keys):
        ctx = self._context(device, dtype)
        slot_index = ctx.block_to_slot.get(block_index)
        if slot_index is None:
            slot_index = block_index & 1
            self._schedule(ctx, slot_index, block_index, dtype, keys)
        else:
            slot = ctx.slots[slot_index]
            if keys is not None and not set(keys).issubset(slot.tensors):
                self._schedule(ctx, slot_index, block_index, dtype, keys)
        slot = ctx.slots[slot_index]
        if slot.ready is not None:
            torch.cuda.current_stream(device).wait_event(slot.ready)
        source = self._source(block_index, keys)
        result = {key: slot.tensors[key] for key in source}
        if keys is None and block_index + 1 < len(self):
            next_slot = 1 - slot_index
            if ctx.slots[next_slot].block != block_index + 1:
                self._schedule(ctx, next_slot, block_index + 1, dtype, None)
        return result

    def weights_on(
        self,
        block_index: int,
        device: torch.device | str,
        dtype: torch.dtype,
        keys: Collection[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._closed:
            raise RuntimeError("VDN-H3 branch weights were released permanently")
        device = torch.device(device)
        if self.mode == "stream" and device.type == "cuda":
            return self._stream_weights(block_index, device, dtype, keys)
        source = self._source(block_index, keys)
        return {
            key: tensor
            if tensor.device == device and tensor.dtype == self._target_dtype(key, dtype)
            else tensor.to(
                device=device,
                dtype=self._target_dtype(key, dtype),
                non_blocking=(device.type == "cuda" and tensor.device.type == "cpu" and tensor.is_pinned()),
            )
            for key, tensor in source.items()
        }

    def mark_consumed(self, block_index, device, dtype):
        if self.mode != "stream":
            return
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        ctx = self._contexts.get((device.type, device.index, dtype))
        if ctx is None:
            return
        slot_index = ctx.block_to_slot.get(block_index)
        if slot_index is None:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        ctx.slots[slot_index].consumed = event

    def release(self) -> "ManagedBranchWeights":
        self._contexts.clear()
        if self.mode == "resident":
            unloaded = False
            if self._patcher is not None:
                try:
                    import comfy.model_management as model_management
                    model_management.unload_model_and_clones(
                        self._patcher, unload_additional_models=False
                    )
                    unloaded = True
                except Exception:
                    unloaded = False
            if not unloaded:
                self.model.to(device="cpu")
        return self

    def close(self) -> None:
        if self._closed:
            return
        self.release()
        for block in self.model.blocks:
            for name in list(block._parameters):
                block._parameters[name] = None
            block._names.clear()
        self._closed = True


__all__ = ["BranchWeightsModel", "FP32_BRANCH_KEYS", "ManagedBranchWeights"]
