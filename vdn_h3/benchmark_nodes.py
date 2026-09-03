"""Small ComfyUI nodes for apples-to-apples sampler timing.

Place ``Kirei Benchmark Start`` immediately before the sampler's MODEL input and connect
the sampler LATENT directly to ``Kirei Benchmark End.after``. The start node synchronizes
the selected CUDA device before opening the timing interval; the end node synchronizes
before closing it. The resulting wall time therefore measures the same graph segment for
native, Turbo and VDN paths instead of relying on UI timestamps.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import torch


_TOKEN_TYPE = "KIREI_BENCHMARK_TOKEN"


def _model_device(model: Any) -> torch.device:
    try:
        return torch.device(getattr(model, "load_device", "cpu"))
    except Exception:
        return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _cuda_start(device: torch.device, reset_peak: bool) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_start_bytes": None,
            "reserved_start_bytes": None,
        }
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_start_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_start_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _cuda_end(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_end_bytes": None,
            "reserved_end_bytes": None,
            "peak_vram_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_end_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_end_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


class KireiBenchmarkStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scenario_id": (
                    "STRING",
                    {
                        "default": "benchmark",
                        "tooltip": "Use an id from benchmarks/scenarios.json.",
                    },
                ),
            },
            "optional": {
                "reset_peak_vram": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", _TOKEN_TYPE)
    RETURN_NAMES = ("model", "benchmark_token")
    FUNCTION = "start"
    CATEGORY = "model_patches/video/benchmark"
    DESCRIPTION = (
        "Synchronize the model device and start a sampler benchmark interval. Connect the "
        "MODEL output directly to the sampler."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Always execute: benchmark nodes must never be reused from Comfy's output cache.
        return float("nan")

    def start(self, model, scenario_id, reset_peak_vram=True):
        device = _model_device(model)
        _sync(device)
        memory = _cuda_start(device, bool(reset_peak_vram))
        token = {
            "scenario_id": str(scenario_id),
            "device": str(device),
            "started_ns": time.perf_counter_ns(),
            "reset_peak_vram": bool(reset_peak_vram),
            **memory,
        }
        return model, token


class KireiBenchmarkEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "benchmark_token": (_TOKEN_TYPE,),
                "after": (
                    "*",
                    {
                        "tooltip": (
                            "Connect the sampler LATENT directly here. Connecting VAE/video output "
                            "would intentionally include decode/encoding in the measured interval."
                        )
                    },
                ),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Optional patched model. When it carries VDN state, the measurement "
                            "embeds the Kirei Runtime Report."
                        )
                    },
                ),
                "run_kind": (["cold", "warm"], {"default": "warm"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("measurement_json",)
    FUNCTION = "end"
    CATEGORY = "model_patches/video/benchmark"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Close the sampler benchmark after GPU synchronization and emit wall time, peak "
        "VRAM and optional VDN runtime metadata as JSON."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def end(self, benchmark_token, after, model=None, run_kind="warm"):
        del after
        if not isinstance(benchmark_token, dict) or "started_ns" not in benchmark_token:
            raise TypeError("Kirei Benchmark End received an invalid benchmark token")
        device = torch.device(benchmark_token.get("device", "cpu"))
        _sync(device)
        finished_ns = time.perf_counter_ns()
        elapsed = (finished_ns - int(benchmark_token["started_ns"])) / 1_000_000_000.0
        payload = {
            "scenario_id": str(benchmark_token.get("scenario_id", "benchmark")),
            "run_kind": str(run_kind),
            "device": str(device),
            "sampler_seconds": elapsed,
            "started_ns": int(benchmark_token["started_ns"]),
            "finished_ns": int(finished_ns),
            "reset_peak_vram": bool(benchmark_token.get("reset_peak_vram", False)),
            "allocated_start_bytes": benchmark_token.get("allocated_start_bytes"),
            "reserved_start_bytes": benchmark_token.get("reserved_start_bytes"),
            **_cuda_end(device),
        }
        if model is not None:
            try:
                from .benchmark import runtime_snapshot

                payload["runtime_report"] = runtime_snapshot(model)
            except RuntimeError:
                # Native/Turbo control models intentionally carry no VDN state.
                payload["runtime_report"] = None
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "KireiBenchmarkStart": KireiBenchmarkStart,
    "KireiBenchmarkEnd": KireiBenchmarkEnd,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiBenchmarkStart": "Kirei Benchmark Start",
    "KireiBenchmarkEnd": "Kirei Benchmark End",
}


__all__ = [
    "KireiBenchmarkStart",
    "KireiBenchmarkEnd",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
