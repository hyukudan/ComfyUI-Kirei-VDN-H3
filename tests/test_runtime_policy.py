from types import SimpleNamespace
import json

import torch

from vdn_h3 import nodes
from vdn_h3.benchmark import runtime_snapshot
from vdn_h3.report_node import KireiVDNH3RuntimeReport


class _Diagnostics:
    enabled = True

    def snapshot(self, flush=True):
        return {"attention.softmax": {"last_ms": 1.25}, "flush": flush}


class _WeightStore:
    mode = "resident"
    nbytes = 4 * 1024**3

    def telemetry(self):
        return {"mode": self.mode, "total_bytes": self.nbytes, "h2d_bytes": 123}


class _Calibration:
    def snapshot(self):
        return {"path": "fixture.json", "entries": 1}


class _Cache:
    _broken = {"flex": "synthetic failure"}
    last_calibration_hit = "grouped"
    calibration = _Calibration()


class _Patcher:
    load_device = torch.device("cpu")

    def __init__(self):
        state = SimpleNamespace(
            name="fixture",
            forwards=3,
            weight_mode="resident",
            weight_store=_WeightStore(),
            attention_backend="auto",
            last_attention_backend="grouped",
            window_cache=_Cache(),
            kernel_backend="auto",
            linear_kernels="auto",
            compile_policy="shared",
            tile_frames=5,
            projection_precision="bf16",
            projection_info=None,
            lora_runtime=None,
            curve_adapter=None,
            inference=True,
            diagnostics=_Diagnostics(),
        )
        self.object_patches = {"diffusion_model._vdn_h3_state": state}


def _branch_maps():
    return [
        {
            "to_out_linear.weight": torch.zeros(8, 8),
            "small": torch.zeros(16),
        }
    ]


def test_profile_resolution_has_exact_reference_and_domestic_low_vram(monkeypatch):
    monkeypatch.setattr(nodes, "_auto_branch_mode", lambda *args: "resident")
    monkeypatch.setattr(nodes, "_auto_tile_frames", lambda *args: 0)
    model = object()

    reference = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="reference",
        linear_kernels="auto",
    )
    assert reference == {
        "profile": "reference",
        "branch_mode": "stream",
        "lora_mode": "bypass",
        "attention_backend": "reference",
        "kernel_backend": "eager",
        "compile_policy": "off",
        "tile_frames": 0,
        "pin_strategy": "auto",
        "projection_precision": "bf16",
        "inference": False,
    }

    compat = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="compat_reference",
    )
    assert compat["attention_backend"] == "compat"
    assert compat["compile_policy"] == "off"
    assert compat["projection_precision"] == "bf16"

    low = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="low_vram",
    )
    assert low["branch_mode"] == "hybrid"
    assert low["tile_frames"] == 5
    assert low["lora_mode"] == "bypass"
    assert low["projection_precision"] == "bf16"
    assert low["inference"]

    fast = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="max_speed",
    )
    assert fast["branch_mode"] == "resident"
    assert fast["compile_policy"] == "reduce_overhead"
    assert fast["tile_frames"] == 0

    fp8 = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="experimental_fp8",
    )
    assert fp8["projection_precision"] == "fp8"
    assert fp8["inference"]


def test_reference_rejects_fp8_precision():
    import pytest

    with pytest.raises(ValueError, match="cannot be used as a reference"):
        nodes._resolve_runtime(
            object(), _branch_maps(), profile="reference", projection_precision="fp8"
        )


def test_auto_branch_mode_separates_24_and_96_gib(monkeypatch):
    gib = 1024**3
    model = SimpleNamespace(load_device=torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nodes, "_model_cuda_device", lambda _: torch.device("cuda:0"))

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=24 * gib),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * gib, 24 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib, int(4.5 * gib)) == "hybrid"
    assert nodes._auto_tile_frames(model, "hybrid") == 5

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=96 * gib),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (70 * gib, 96 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib, int(4.5 * gib)) == "resident"
    assert nodes._auto_tile_frames(model, "resident") == 0

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (5 * gib, 96 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib, int(4.5 * gib)) == "stream"


def test_runtime_snapshot_and_node_are_machine_readable():
    model = _Patcher()
    snapshot = runtime_snapshot(model)
    assert snapshot["checkpoint"] == "fixture"
    assert snapshot["branch_gib"] == 4.0
    assert snapshot["branch_storage"]["h2d_bytes"] == 123
    assert snapshot["attention_failures"] == {"flex": "synthetic failure"}
    assert snapshot["attention_calibration"]["last_hit"] == "grouped"
    assert snapshot["compile_policy"] == "shared"
    assert snapshot["tile_frames"] == 5
    assert snapshot["cuda"]["available"] is False

    node = KireiVDNH3RuntimeReport()
    sentinel = object()
    passthrough, text = node.report(model, after=sentinel)
    assert passthrough is model
    parsed = json.loads(text)
    assert parsed["attention_last"] == "grouped"
    assert parsed["diagnostics"]["attention.softmax"]["last_ms"] == 1.25
    schema = node.INPUT_TYPES()
    assert schema["optional"]["after"][0] == "*"
    assert node.OUTPUT_NODE is True
