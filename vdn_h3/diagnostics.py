"""Low-overhead, model-owned diagnostics for the Kirei VDN-H3 runtime."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch


_LOG = logging.getLogger("comfy.vdn_h3")


class DiagnosticsRecorder:
    """Collect timing and CUDA-memory samples only when explicitly enabled.

    Timings synchronize CUDA so they are intentionally a debugging facility rather
    than part of the fast path. All state belongs to one patched VDN model. Peak-memory
    counters are reset on the first measured scope for each CUDA device so the report
    describes this VDN run instead of an unrelated earlier allocation peak.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._max_ms: dict[str, float] = defaultdict(float)
        self._last_memory: dict[str, int] = {}
        self._baseline_memory: dict[tuple[str, int | None], dict[str, int]] = {}
        self._seen_devices: set[tuple[str, int | None]] = set()

    @staticmethod
    def _device_key(resolved: torch.device) -> tuple[str, int | None]:
        return resolved.type, resolved.index

    def _prepare_cuda_device(self, resolved: torch.device) -> None:
        key = self._device_key(resolved)
        if key in self._seen_devices:
            return
        torch.cuda.synchronize(resolved)
        allocated = int(torch.cuda.memory_allocated(resolved))
        reserved = int(torch.cuda.memory_reserved(resolved))
        torch.cuda.reset_peak_memory_stats(resolved)
        self._baseline_memory[key] = {
            "allocated": allocated,
            "reserved": reserved,
        }
        self._seen_devices.add(key)

    @contextmanager
    def scope(self, name: str, device: torch.device | str | None = None):
        if not self.enabled:
            yield
            return
        resolved = torch.device(device) if device is not None else None
        cuda = resolved is not None and resolved.type == "cuda" and torch.cuda.is_available()
        if cuda:
            self._prepare_cuda_device(resolved)
            torch.cuda.synchronize(resolved)
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize(resolved)
            elapsed = (time.perf_counter() - started) * 1000.0
            self._totals[name] += elapsed
            self._counts[name] += 1
            self._max_ms[name] = max(self._max_ms[name], elapsed)
            if cuda:
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

    def snapshot(self) -> dict[str, Any]:
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
            "stages": stages,
            "cuda_memory": dict(self._last_memory),
        }

    def log(self, *, prefix: str = "VDN-H3 diagnostics") -> None:
        if not self.enabled:
            return
        snap = self.snapshot()
        compact = ", ".join(
            f"{name}={values['mean_ms']:.2f}ms"
            for name, values in snap["stages"].items()
        )
        memory = snap["cuda_memory"]
        if memory:
            gib = 1024**3
            compact += (
                f"; device={memory.get('device', '?')}"
                f" alloc={memory['allocated']/gib:.2f}GiB"
                f" reserved={memory['reserved']/gib:.2f}GiB"
                f" peak={memory['peak_allocated']/gib:.2f}GiB"
                f" peak_delta={memory['peak_allocated_delta']/gib:.2f}GiB"
            )
        _LOG.info("%s: %s", prefix, compact or "no samples")

    def clear(self) -> None:
        self._totals.clear()
        self._counts.clear()
        self._max_ms.clear()
        self._last_memory.clear()
        self._baseline_memory.clear()
        self._seen_devices.clear()


__all__ = ["DiagnosticsRecorder"]
