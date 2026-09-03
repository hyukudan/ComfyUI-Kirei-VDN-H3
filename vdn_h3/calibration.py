"""Persistent per-GPU attention calibration for VDN-H3.

Calibration is explicit: normal generation never launches hidden benchmarks. The auto
backend consults this store when an exact hardware/geometry entry exists, otherwise it
uses conservative built-in heuristics.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


CALIBRATION_VERSION = 3
CALIBRATABLE_BACKENDS = ("grouped", "flex", "flash2", "decomposed")


def _package_version(distribution: str, module: str | None = None) -> str | None:
    try:
        from importlib.metadata import version

        return str(version(distribution))
    except Exception:
        pass
    if module:
        try:
            import importlib

            loaded = importlib.import_module(module)
            return str(getattr(loaded, "__version__", None) or "present")
        except Exception:
            return None
    return None


def _driver_version() -> str | None:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            value = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        return value.decode() if isinstance(value, bytes) else str(value)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def runtime_environment() -> dict[str, Any]:
    """Software identity that decides which exact backends exist and how fast they run.

    Cached per process and embedded in every calibration signature: a node update, a
    torch/CUDA/driver change or a newly installed flash-attn 2 / flash-attn-4 / Triton
    changes the signature, so old winners are re-measured instead of trusted.
    """
    from . import __version__
    from .window import backend_inventory

    cudnn = torch.backends.cudnn
    return {
        "node": __version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": cudnn.version() if cudnn.is_available() else None,
        "driver": _driver_version() if torch.cuda.is_available() else None,
        "triton": _package_version("triton", "triton"),
        "flash_attn": _package_version("flash-attn", "flash_attn"),
        "flash_attn_4": _package_version("flash-attn-4", "flash_attn.cute"),
        "backends": [name for name, ok in sorted(backend_inventory(None).items()) if ok],
    }


def default_calibration_path() -> Path:
    try:
        import folder_paths

        roots = folder_paths.get_folder_paths("vdn")
        if roots:
            return Path(roots[0]) / "vdn_h3_calibration.json"
    except Exception:
        pass
    return Path.cwd() / "models" / "vdn" / "vdn_h3_calibration.json"


def calibration_signature(
    query: torch.Tensor,
    num_frames: int,
    bounds,
    anchor_frames: str,
    *,
    groups: int,
    video_start: int | None = None,
    video_end: int | None = None,
    tokens_per_frame: int | None = None,
) -> str:
    device = query.device
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu = props.name
        capability = list(torch.cuda.get_device_capability(device))
    else:
        gpu = str(device)
        capability = []
    payload = {
        "version": CALIBRATION_VERSION,
        "env": runtime_environment(),
        "gpu": gpu,
        "capability": capability,
        "dtype": str(query.dtype),
        "sequence": int(query.shape[0]),
        "heads": int(query.shape[1]),
        "head_dim": int(query.shape[2]),
        "frames": int(num_frames),
        "groups": int(groups),
        "video_start": None if video_start is None else int(video_start),
        "video_end": None if video_end is None else int(video_end),
        "tokens_per_frame": None if tokens_per_frame is None else int(tokens_per_frame),
        "anchor_frames": str(anchor_frames),
        "bounds": [[int(lo), int(hi)] for lo, hi in bounds],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class CalibrationStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_calibration_path()
        self._loaded = False
        self._entries: dict[str, dict[str, Any]] = {}

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            return
        if data.get("version") != CALIBRATION_VERSION:
            return
        entries = data.get("entries")
        if isinstance(entries, dict):
            self._entries = {
                str(key): value for key, value in entries.items() if isinstance(value, dict)
            }

    def lookup(self, signature: str) -> str | None:
        self._load()
        entry = self._entries.get(signature)
        if not entry:
            return None
        backend = entry.get("winner")
        return backend if backend in CALIBRATABLE_BACKENDS else None

    def record(self, signature: str, *, winner: str, results: dict[str, Any]):
        if winner not in CALIBRATABLE_BACKENDS:
            raise ValueError(f"cannot calibrate unsupported backend {winner!r}")
        self._load()
        self._entries[signature] = {"winner": winner, "results": results}

    def save(self):
        self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CALIBRATION_VERSION, "entries": self._entries}
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def snapshot(self):
        self._load()
        return {
            "path": str(self.path),
            "version": CALIBRATION_VERSION,
            "entries": len(self._entries),
            "environment": runtime_environment(),
        }


__all__ = [
    "CALIBRATABLE_BACKENDS",
    "CALIBRATION_VERSION",
    "CalibrationStore",
    "calibration_signature",
    "default_calibration_path",
    "runtime_environment",
]
