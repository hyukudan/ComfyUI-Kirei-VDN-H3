"""Persistent per-GPU attention calibration for VDN-H3.

Calibration is explicit: normal generation never launches hidden benchmarks. The auto
backend consults this store when an exact hardware/geometry entry exists, otherwise it
uses conservative built-in heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


CALIBRATION_VERSION = 1
CALIBRATABLE_BACKENDS = ("grouped", "flex", "flash2", "decomposed")


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
        "torch": torch.__version__,
        "gpu": gpu,
        "capability": capability,
        "dtype": str(query.dtype),
        "sequence": int(query.shape[0]),
        "heads": int(query.shape[1]),
        "head_dim": int(query.shape[2]),
        "frames": int(num_frames),
        "groups": int(groups),
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
        payload = {
            "version": CALIBRATION_VERSION,
            "entries": self._entries,
        }
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def snapshot(self):
        self._load()
        return {"path": str(self.path), "entries": len(self._entries)}


__all__ = [
    "CALIBRATABLE_BACKENDS",
    "CALIBRATION_VERSION",
    "CalibrationStore",
    "calibration_signature",
    "default_calibration_path",
]
