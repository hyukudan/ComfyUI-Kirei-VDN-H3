"""Model-owned diagnostics that do not perturb the normal CUDA schedule."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


_LOG = logging.getLogger("comfy.vdn_h3")


@dataclass
class _CudaSample:
    name: str
    start: Any
    end: Any
    device: torch.device


class DiagnosticsRecorder:
    """Collect stage timings and memory only when explicitly enabled.

    CUDA scopes use events recorded on the current stream and therefore do not insert
    synchronizations into the render. Pending events are resolved once when a snapshot
    or explicit log is requested. This preserves transfer/compute overlap while still
    producing accurate stage timings after the render.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._max_ms: dict[str, float] = defaultdict(float)
        self._pending: list[_CudaSample] = []
        self._last_memory: dict[str, int | str] = {}
        self._baseline_memory: dict[tuple[str, int | None], dict[str, int]] = {}
        self._seen_devices: set[tuple[str, int | None]] = set()

    @staticmethod
    def _device_key(resolved: torch.device) -> tuple[str, int | None]:
        return resolved.type, resolved.index

    def _prepare_cuda_device(self, resolved: torch.device) -> None:
        key = self._device_key(resolved)
        if key in self._seen_devices:
            return
        allocated = int(torch.cuda.memory_allocated(resolved))
        reserved = int(torch.cuda.memory_reserved(resolved))
        torch.cuda.reset_peak_memory_stats(resolved)
        self._baseline_memory[key] = {"allocated": allocated, "reserved": reserved}
        self._seen_devices.add(key)

    def _record_elapsed(self, name: str, elapsed: float) -> None:
        self._totals[name] += elapsed
        self._counts[name] += 1
        self._max_ms[name] = max(self._max_ms[name], elapsed)

    def _sample_memory(self, resolved: torch.device) -> None:
        key = self._device_key(resolved)
        baseline = self._baseline_memory.get(key, {})
        peak_allocated = int(torch.cuda.max_memory_allocated(resolved))
        peak_reserved = int(torch.cuda.max_memory_reserved(resolved))
        baseline_allocated = int(baseline.get("allocated", 0))
        baseline_reserved = int(baseline.get("reserved", 0))
        self._last_memory = {
            "device": str(resolved),
            "baseline_allocated": baseline_allocated,
            "baseline_reserved": baseline_reserved,
            "allocated": int(torch.cuda.memory_allocated(resolved)),
            "reserved": int(torch.cuda.memory_reserved(resolved)),
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "peak_allocated_delta": max(0, peak_allocated - baseline_allocated),
            "peak_reserved_delta": max(0, peak_reserved - baseline_reserved),
        }

    @contextmanager
    def scope(self, name: str, device: torch.device | str | None = None):
        if not self.enabled:
            yield
            return
        resolved = torch.device(device) if device is not None else None
        cuda = resolved is not None and resolved.type == "cuda" and torch.cuda.is_available()
        if not cuda:
            started = time.perf_counter()
            try:
                yield
            finally:
                self._record_elapsed(name, (time.perf_counter() - started) * 1000.0)
            return

        self._prepare_cuda_device(resolved)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(resolved)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self._pending.append(_CudaSample(name, start, end, resolved))
            # Allocation counters are host-side queries and do not synchronize kernels.
            self._sample_memory(resolved)

    def flush(self) -> None:
        """Resolve pending CUDA events with at most one synchronization per device."""
        if not self._pending:
            return
        devices = {(sample.device.type, sample.device.index): sample.device for sample in self._pending}
        for device in devices.values():
            torch.cuda.synchronize(device)
        pending, self._pending = self._pending, []
        for sample in pending:
            try:
                elapsed = float(sample.start.elapsed_time(sample.end))
            except Exception:
                continue
            self._record_elapsed(sample.name, elapsed)
        for device in devices.values():
            self._sample_memory(device)

    def snapshot(self, *, flush: bool = True) -> dict[str, Any]:
        if flush and self.enabled:
            self.flush()
        stages = {}
        for name in sorted(self._counts):
            count = self._counts[name]
            stages[name] = {
                "count": count,
                "total_ms": self._totals[name],
                "mean_ms": self._totals[name] / count,
                "max_ms": self._max_ms[name],
            }
        return {
            "enabled": self.enabled,
            "pending_cuda_samples": len(self._pending),
            "stages": stages,
            "cuda_memory": dict(self._last_memory),
        }

    def log(self, *, prefix: str = "VDN-H3 diagnostics") -> None:
        if not self.enabled:
            return
        snap = self.snapshot(flush=True)
        compact = ", ".join(
            f"{name}={values['mean_ms']:.2f}ms"
            for name, values in snap["stages"].items()
        )
        memory = snap["cuda_memory"]
        if memory:
            gib = 1024**3
            compact += (
                f"; device={memory.get('device', '?')}"
                f" alloc={int(memory['allocated'])/gib:.2f}GiB"
                f" reserved={int(memory['reserved'])/gib:.2f}GiB"
                f" peak={int(memory['peak_allocated'])/gib:.2f}GiB"
                f" peak_delta={int(memory['peak_allocated_delta'])/gib:.2f}GiB"
            )
        _LOG.info("%s: %s", prefix, compact or "no samples")

    def clear(self) -> None:
        self._totals.clear()
        self._counts.clear()
        self._max_ms.clear()
        self._pending.clear()
        self._last_memory.clear()
        self._baseline_memory.clear()
        self._seen_devices.clear()


__all__ = ["DiagnosticsRecorder"]
