"""ComfyUI-accounted VDN branch storage with resident, streamed and hybrid modes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .auxiliary import create_auxiliary_patcher, unload_auxiliary


FP32_BRANCH_KEYS = frozenset(
    {"alpha.A_log", "alpha.dt_bias", "to_out_linear.weight_scale"}
)
PRESERVE_DTYPE_KEYS = frozenset({"to_out_linear.weight_fp8"})
STREAMED_PROJECTION_KEY = "to_out_linear.weight"
FP8_STREAMED_PROJECTION_KEY = "to_out_linear.weight_fp8"
BRANCH_MODES = ("resident", "hybrid", "stream")
PIN_STRATEGIES = ("auto", "comfy", "all", "none")


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
    def __init__(self, weights: Sequence[Mapping[str, torch.Tensor]]):
        super().__init__()
        if not weights:
            raise ValueError("VDN branch weights cannot be empty")
        self.blocks = nn.ModuleList(
            [_WeightBlock(block, index) for index, block in enumerate(weights)]
        )
        self.device = torch.device("cpu")

    def block(self, index: int) -> _WeightBlock:
        return self.blocks[index]

    def forward(self, *args, **kwargs):
        raise RuntimeError("BranchWeightsModel is storage-only and has no forward")


class _WeightContainer(nn.Module):
    def __init__(self, resident: BranchWeightsModel | None, streamed: BranchWeightsModel | None):
        super().__init__()
        self.resident = resident
        self.streamed = streamed
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        raise RuntimeError("VDN branch weight container is storage-only")


@dataclass
class _StreamSlot:
    tensors: dict[str, torch.Tensor]
    block: int | None = None
    valid_keys: set[str] = field(default_factory=set)
    ready: Any = None
    consumed: Any = None
    ready_recorded: bool = False
    consumed_recorded: bool = False


class _StreamContext:
    def __init__(self, device: torch.device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device):
            self.slots = [
                _StreamSlot(
                    {},
                    ready=torch.cuda.Event(blocking=False),
                    consumed=torch.cuda.Event(blocking=False),
                )
                for _ in range(2)
            ]
        self.block_to_slot: dict[int, int] = {}


class ManagedBranchWeights(nn.Module):
    """Own branch tensors with safe resident, stream and hybrid placement."""

    def __init__(
        self,
        weights: Sequence[Mapping[str, torch.Tensor]],
        mode: str = "stream",
        *,
        pin_strategy: str = "auto",
        streamed_keys: Collection[str] = (STREAMED_PROJECTION_KEY,),
    ):
        super().__init__()
        if mode not in BRANCH_MODES:
            raise ValueError(f"branch weight mode must be one of {BRANCH_MODES}, got {mode!r}")
        if pin_strategy not in PIN_STRATEGIES:
            raise ValueError(f"pin strategy must be one of {PIN_STRATEGIES}, got {pin_strategy!r}")
        self.mode = mode
        self.pin_strategy = pin_strategy
        self.streamed_keys = frozenset(str(key) for key in streamed_keys)
        if mode == "hybrid" and not self.streamed_keys:
            raise ValueError("hybrid branch mode needs at least one streamed key")

        resident_maps = streamed_maps = None
        if mode == "resident":
            resident_maps = [dict(block) for block in weights]
        elif mode == "stream":
            streamed_maps = [dict(block) for block in weights]
        else:
            resident_maps, streamed_maps = [], []
            for index, block in enumerate(weights):
                streamed = {k: v for k, v in block.items() if k in self.streamed_keys}
                resident = {k: v for k, v in block.items() if k not in self.streamed_keys}
                missing = self.streamed_keys - set(streamed)
                if missing:
                    raise KeyError(
                        f"VDN block {index} is missing hybrid streamed keys {sorted(missing)}"
                    )
                resident_maps.append(resident)
                streamed_maps.append(streamed)

        resident_model = BranchWeightsModel(resident_maps) if resident_maps is not None else None
        streamed_model = BranchWeightsModel(streamed_maps) if streamed_maps is not None else None
        self.model = _WeightContainer(resident_model, streamed_model)
        self._closed = False
        self._contexts: dict[tuple[str, int | None, torch.dtype], _StreamContext] = {}
        self._patchers: dict[str, Any] = {}
        self._stats: dict[str, int] = defaultdict(int)
        self._pinned_bytes = 0
        if streamed_model is not None:
            self._pin_cpu_weights(streamed_model)

    def __len__(self) -> int:
        model = self.model.resident or self.model.streamed
        return len(model.blocks)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    @property
    def resident_nbytes(self) -> int:
        model = self.model.resident
        return 0 if model is None else sum(p.numel() * p.element_size() for p in model.parameters())

    @property
    def streamed_nbytes(self) -> int:
        model = self.model.streamed
        return 0 if model is None else sum(p.numel() * p.element_size() for p in model.parameters())

    def _pin_cpu_weights(self, model: BranchWeightsModel) -> None:
        if self.pin_strategy == "none":
            return
        try:
            import comfy.model_management as model_management
        except ImportError:
            model_management = None
        for param in model.parameters():
            if param.device.type != "cpu" or param.is_pinned():
                if param.is_pinned():
                    self._pinned_bytes += param.numel() * param.element_size()
                continue
            pinned = False
            if model_management is not None and self.pin_strategy in {"auto", "comfy", "all"}:
                try:
                    pinned = bool(model_management.pin_memory(param.data))
                except Exception:
                    pinned = False
            if not pinned and self.pin_strategy == "all" and torch.cuda.is_available():
                try:
                    param.data = param.data.pin_memory()
                    pinned = bool(param.is_pinned())
                except Exception:
                    pinned = False
            if pinned or param.is_pinned():
                self._pinned_bytes += param.numel() * param.element_size()

    def attach_to(self, base_patcher: Any, key: str = "vdn_h3_branch"):
        try:
            if self.model.resident is not None and "resident" not in self._patchers:
                self._patchers["resident"] = create_auxiliary_patcher(
                    self.model.resident,
                    base_patcher,
                    size=self.resident_nbytes,
                    label="VDN resident branch weights",
                    load_device=getattr(base_patcher, "load_device", torch.device("cpu")),
                    offload_device=getattr(base_patcher, "offload_device", torch.device("cpu")),
                )
            if self.model.streamed is not None and "streamed" not in self._patchers:
                self._patchers["streamed"] = create_auxiliary_patcher(
                    self.model.streamed,
                    base_patcher,
                    size=self.streamed_nbytes,
                    label="VDN streamed branch weights",
                    load_device=torch.device("cpu"),
                    offload_device=torch.device("cpu"),
                )
        except ImportError:
            return None
        setter = getattr(base_patcher, "set_additional_models", None)
        if setter is not None:
            setter(key, list(self._patchers.values()))
        return tuple(self._patchers.values())

    @staticmethod
    def _target_dtype(
        key: str,
        compute_dtype: torch.dtype,
        source_dtype: torch.dtype | None = None,
    ) -> torch.dtype:
        if key in FP32_BRANCH_KEYS:
            return torch.float32
        if key in PRESERVE_DTYPE_KEYS and source_dtype is not None:
            return source_dtype
        return compute_dtype

    @staticmethod
    def _source_from(
        model: BranchWeightsModel | None,
        block_index: int,
        keys: Collection[str] | None,
    ) -> dict[str, torch.Tensor]:
        if model is None:
            return {}
        block = model.block(block_index)
        wanted = tuple(block.keys()) if keys is None else tuple(sorted(set(keys)))
        present = set(block.keys())
        wanted = tuple(key for key in wanted if key in present)
        return {key: block.tensor(key) for key in wanted}

    def _all_keys(self, block_index: int) -> set[str]:
        keys: set[str] = set()
        for model in (self.model.resident, self.model.streamed):
            if model is not None:
                keys.update(model.block(block_index).keys())
        return keys

    def _partition_keys(self, block_index: int, keys: Collection[str] | None):
        wanted = self._all_keys(block_index) if keys is None else set(keys)
        missing = wanted - self._all_keys(block_index)
        if missing:
            raise KeyError(
                f"VDN-H3 branch block {block_index} is missing requested weights {sorted(missing)}"
            )
        resident_keys = set()
        streamed_keys = set()
        if self.model.resident is not None:
            resident_keys = wanted & set(self.model.resident.block(block_index).keys())
        if self.model.streamed is not None:
            streamed_keys = wanted & set(self.model.streamed.block(block_index).keys())
        return resident_keys, streamed_keys

    def _context(self, device: torch.device, dtype: torch.dtype) -> _StreamContext:
        key = (device.type, device.index, dtype)
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = _StreamContext(device)
            self._contexts[key] = ctx
        return ctx

    def _schedule(
        self,
        ctx: _StreamContext,
        slot_index: int,
        block_index: int,
        compute_dtype: torch.dtype,
        keys: Collection[str],
        *,
        prefetch: bool,
    ) -> None:
        if not keys:
            return
        slot = ctx.slots[slot_index]
        source = self._source_from(self.model.streamed, block_index, keys)
        required = set(source)
        if slot.block == block_index and required.issubset(slot.valid_keys):
            return
        if slot.consumed_recorded:
            ctx.stream.wait_event(slot.consumed)
        switching = slot.block != block_index
        if switching:
            if slot.block is not None:
                ctx.block_to_slot.pop(slot.block, None)
            slot.valid_keys.clear()
            slot.block = block_index
            ctx.block_to_slot[block_index] = slot_index
        with torch.cuda.stream(ctx.stream):
            for key, tensor in source.items():
                target_dtype = self._target_dtype(key, compute_dtype, tensor.dtype)
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
                self._stats["h2d_bytes"] += int(target.numel() * target.element_size())
                self._stats["copies"] += 1
            slot.ready.record(ctx.stream)
        slot.valid_keys.update(required)
        slot.ready_recorded = True
        slot.consumed_recorded = False
        self._stats["scheduled_blocks"] += 1
        if prefetch:
            self._stats["prefetch_blocks"] += 1

    def prefetch(
        self,
        block_index: int,
        device: torch.device | str,
        dtype: torch.dtype,
        keys: Collection[str] | None = None,
    ) -> None:
        if self._closed or self.model.streamed is None:
            return
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        _, streamed_keys = self._partition_keys(block_index, keys)
        if not streamed_keys:
            return
        ctx = self._context(device, dtype)
        slot_index = ctx.block_to_slot.get(block_index, block_index & 1)
        self._schedule(
            ctx, slot_index, block_index, dtype, streamed_keys, prefetch=True
        )

    def _stream_weights(
        self,
        block_index: int,
        device: torch.device,
        dtype: torch.dtype,
        keys: Collection[str],
    ) -> dict[str, torch.Tensor]:
        if not keys:
            return {}
        ctx = self._context(device, dtype)
        slot_index = ctx.block_to_slot.get(block_index)
        if slot_index is None:
            slot_index = block_index & 1
            self._schedule(ctx, slot_index, block_index, dtype, keys, prefetch=False)
        else:
            slot = ctx.slots[slot_index]
            required = set(keys)
            if not required.issubset(slot.valid_keys):
                self._schedule(ctx, slot_index, block_index, dtype, keys, prefetch=False)
        slot = ctx.slots[slot_index]
        if slot.ready_recorded:
            torch.cuda.current_stream(device).wait_event(slot.ready)
            self._stats["ready_wait_events"] += 1
        self._stats["served_stream_requests"] += 1
        return {key: slot.tensors[key] for key in keys}

    def _resident_weights(
        self,
        block_index: int,
        device: torch.device,
        dtype: torch.dtype,
        keys: Collection[str],
    ) -> dict[str, torch.Tensor]:
        source = self._source_from(self.model.resident, block_index, keys)
        result = {}
        for key, tensor in source.items():
            target_dtype = self._target_dtype(key, dtype, tensor.dtype)
            result[key] = (
                tensor
                if tensor.device == device and tensor.dtype == target_dtype
                else tensor.to(
                    device=device,
                    dtype=target_dtype,
                    non_blocking=(
                        device.type == "cuda" and tensor.device.type == "cpu" and tensor.is_pinned()
                    ),
                )
            )
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
        if not 0 <= block_index < len(self):
            raise IndexError(f"VDN-H3 branch block {block_index} is out of range")
        device = torch.device(device)
        resident_keys, streamed_keys = self._partition_keys(block_index, keys)
        result = self._resident_weights(block_index, device, dtype, resident_keys)
        if streamed_keys:
            if device.type == "cuda":
                result.update(self._stream_weights(block_index, device, dtype, streamed_keys))
            else:
                source = self._source_from(self.model.streamed, block_index, streamed_keys)
                for key, tensor in source.items():
                    target_dtype = self._target_dtype(key, dtype, tensor.dtype)
                    result[key] = tensor if tensor.dtype == target_dtype else tensor.to(dtype=target_dtype)

        if keys is None and self.model.streamed is not None and len(self) > 1 and device.type == "cuda":
            self.prefetch((block_index + 1) % len(self), device, dtype, None)
        return result

    def mark_consumed(
        self,
        block_index: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if self.model.streamed is None:
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
        slot = ctx.slots[slot_index]
        slot.consumed.record(torch.cuda.current_stream(device))
        slot.consumed_recorded = True

    def telemetry(self) -> dict[str, Any]:
        buffers = 0
        valid_keys = 0
        for ctx in self._contexts.values():
            for slot in ctx.slots:
                buffers += sum(t.numel() * t.element_size() for t in slot.tensors.values())
                valid_keys += len(slot.valid_keys)
        return {
            "mode": self.mode,
            "total_bytes": self.nbytes,
            "resident_bytes": self.resident_nbytes,
            "streamed_bytes": self.streamed_nbytes,
            "pinned_cpu_bytes": int(self._pinned_bytes),
            "gpu_stream_buffer_bytes": int(buffers),
            "valid_stream_keys": int(valid_keys),
            **{key: int(value) for key, value in sorted(self._stats.items())},
        }

    def release(self) -> "ManagedBranchWeights":
        for ctx in self._contexts.values():
            try:
                ctx.stream.synchronize()
            except Exception:
                pass
        self._contexts.clear()
        if self.model.resident is not None:
            unload_auxiliary(self._patchers.get("resident"), self.model.resident)
        if self.model.streamed is not None:
            unload_auxiliary(self._patchers.get("streamed"), self.model.streamed)
        return self

    def close(self) -> None:
        if self._closed:
            return
        self.release()
        for model in (self.model.resident, self.model.streamed):
            if model is None:
                continue
            for block in model.blocks:
                for name in list(block._parameters):
                    block._parameters[name] = None
                block._names.clear()
        self._closed = True


__all__ = [
    "BRANCH_MODES",
    "BranchWeightsModel",
    "FP32_BRANCH_KEYS",
    "FP8_STREAMED_PROJECTION_KEY",
    "ManagedBranchWeights",
    "PIN_STRATEGIES",
    "PRESERVE_DTYPE_KEYS",
    "STREAMED_PROJECTION_KEY",
]
