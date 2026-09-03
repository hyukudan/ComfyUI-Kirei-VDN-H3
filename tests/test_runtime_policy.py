from types import SimpleNamespace
import json

import torch

from vdn_h3 import nodes
from vdn_h3.benchmark import runtime_snapshot
from vdn_h3.report_node import KireiVDNH3RuntimeReport


class _Diagnostics:
    enabled = True

    def snapshot(self):
        return {"attention.softmax": {"last_ms": 1.25}}


class _WeightStore:
    mode = "resident"
    nbytes = 4 * 1024**3


class _Cache:
    _broken = {"flex": "synthetic failure"}


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
            linear_kernels="auto",
            inference=True,
            diagnostics=_Diagnostics(),
        )
        self.object_patches = {"diffusion_model._vdn_h3_state": state}


def _branch_maps():
    return [{"w": torch.zeros(16)}]


def test_profile_resolution_has_safe_reference_and_low_vram_modes(monkeypatch):
    monkeypatch.setattr(nodes, "_auto_branch_mode", lambda model, size: "resident")
    model = object()

    reference = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="reference",
        branch_mode="auto",
        lora_mode="auto",
        attention_backend="auto",
        linear_kernels="auto",
    )
    assert reference == {
        "profile": "reference",
        "branch_mode": "stream",
        "lora_mode": "merge",
        "attention_backend": "reference",
        "linear_kernels": "eager",
        "inference": False,
    }

    low = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="low_vram",
        branch_mode="auto",
        lora_mode="auto",
        attention_backend="auto",
        linear_kernels="auto",
    )
    assert low["branch_mode"] == "stream"
    assert low["lora_mode"] == "bypass"
    assert low["inference"]

    fast = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="max_speed",
        branch_mode="auto",
        lora_mode="auto",
        attention_backend="auto",
        linear_kernels="auto",
    )
    assert fast["branch_mode"] == "resident"
    assert fast["lora_mode"] == "bypass"


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
    assert nodes._auto_branch_mode(model, 5 * gib) == "stream"

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=96 * gib),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (70 * gib, 96 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib) == "resident"

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (18 * gib, 96 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib) == "stream"


def test_runtime_snapshot_and_node_are_machine_readable():
    model = _Patcher()
    snapshot = runtime_snapshot(model)
    assert snapshot["checkpoint"] == "fixture"
    assert snapshot["branch_gib"] == 4.0
    assert snapshot["attention_failures"] == {"flex": "synthetic failure"}
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
