from types import SimpleNamespace
import json

import pytest
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
    last_autotune_error = None
    calibration = _Calibration()


class _Patcher:
    load_device = torch.device("cpu")

    def __init__(self):
        state = SimpleNamespace(
            name="fixture",
            config={},
            adapters={
                "active": ["default", "turbo"],
                "strengths": {"default": 1.0, "turbo": 1.0},
                "lora_mode": "bypass",
            },
            forwards=3,
            profile="auto",
            base_precision="bf16",
            branch_execution="serial",
            block_fusion=False,
            block_fusion_error=None,
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
            last_layout=None,
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
    monkeypatch.setattr(nodes, "_auto_execution_mode", lambda *args: "serial")
    monkeypatch.setattr(nodes, "detect_base_precision", lambda model: "bf16")
    model = object()

    reference = nodes._resolve_runtime(
        model,
        _branch_maps(),
        profile="reference",
        linear_kernels="auto",
    )
    assert reference == {
        "profile": "reference",
        "base_precision": "bf16",
        "branch_mode": "stream",
        "branch_execution": "serial",
        "lora_mode": "bypass",
        "attention_backend": "reference",
        "kernel_backend": "eager",
        "compile_policy": "off",
        "tile_frames": 0,
        "pin_strategy": "auto",
        "projection_precision": "bf16",
        "block_fusion": False,
        "inference": False,
    }

    compat = nodes._resolve_runtime(model, _branch_maps(), profile="compat_reference")
    assert compat["attention_backend"] == "compat"
    assert compat["compile_policy"] == "off"
    assert compat["projection_precision"] == "bf16"
    assert compat["branch_execution"] == "serial"
    assert compat["lora_mode"] == "bypass"
    assert not compat["block_fusion"]

    low = nodes._resolve_runtime(model, _branch_maps(), profile="low_vram")
    assert low["branch_mode"] == "hybrid"
    assert low["branch_execution"] == "serial"
    assert low["tile_frames"] == 5
    assert low["lora_mode"] == "bypass"
    assert low["projection_precision"] == "bf16"
    assert not low["block_fusion"]
    assert low["inference"]

    fast = nodes._resolve_runtime(model, _branch_maps(), profile="max_speed")
    assert fast["branch_mode"] == "resident"
    assert fast["branch_execution"] == "serial"
    assert fast["lora_mode"] == "merge"
    assert fast["compile_policy"] == "reduce_overhead"
    assert fast["tile_frames"] == 0
    assert fast["projection_precision"] == "fp8"
    assert fast["block_fusion"]

    fp8 = nodes._resolve_runtime(model, _branch_maps(), profile="experimental_fp8")
    assert fp8["projection_precision"] == "fp8"
    # Experimental profiles follow auto's adapter rule: a BF16 base merges exactly.
    assert fp8["lora_mode"] == "merge"
    assert fp8["inference"]


def test_auto_int8_resident_is_quality_first_until_quantized_projection_wins(monkeypatch):
    monkeypatch.setattr(nodes, "detect_base_precision", lambda model: "int8")
    monkeypatch.setattr(nodes, "_auto_branch_mode", lambda *args: "resident")
    monkeypatch.setattr(nodes, "_auto_execution_mode", lambda *args: "parallel")
    monkeypatch.setattr(nodes, "_auto_tile_frames", lambda *args: 0)

    runtime = nodes._resolve_runtime(object(), _branch_maps(), profile="auto")
    assert runtime["base_precision"] == "int8"
    assert runtime["branch_mode"] == "resident"
    assert runtime["branch_execution"] == "serial"
    assert runtime["lora_mode"] == "bypass"
    # Current workstation evidence did not show an INT8 VDN projection win, so resident
    # auto stays BF16. INT8/FP8 remain explicit benchmark/max-speed candidates.
    assert runtime["projection_precision"] == "bf16"
    assert runtime["compile_policy"] == "shared"
    assert runtime["tile_frames"] == 0
    assert runtime["block_fusion"]


def test_workstation_fp8_can_opt_into_parallel_and_merge(monkeypatch):
    monkeypatch.setattr(nodes, "detect_base_precision", lambda model: "int8")
    monkeypatch.setattr(nodes, "_auto_execution_mode", lambda *args: "parallel")
    runtime = nodes._resolve_runtime(object(), _branch_maps(), profile="workstation_fp8")
    assert runtime["branch_mode"] == "resident"
    assert runtime["branch_execution"] == "parallel"
    assert runtime["lora_mode"] == "merge"
    assert runtime["projection_precision"] == "fp8"
    assert runtime["compile_policy"] == "shared"


def test_auto_quantized_nonresident_can_reduce_projection_transfer(monkeypatch):
    monkeypatch.setattr(nodes, "detect_base_precision", lambda model: "int8")
    monkeypatch.setattr(nodes, "_auto_branch_mode", lambda *args: "hybrid")
    monkeypatch.setattr(nodes, "_auto_execution_mode", lambda *args: "parallel")
    monkeypatch.setattr(nodes, "_auto_tile_frames", lambda *args: 5)

    runtime = nodes._resolve_runtime(object(), _branch_maps(), profile="auto")
    assert runtime["projection_precision"] == "int8"
    assert runtime["branch_mode"] == "hybrid"
    assert runtime["branch_execution"] == "serial"
    assert runtime["lora_mode"] == "bypass"
    assert runtime["tile_frames"] == 5
    assert not runtime["block_fusion"]


@pytest.mark.parametrize("precision", ["fp8", "int8"])
def test_reference_rejects_quantized_projection(precision):
    with pytest.raises(ValueError, match="cannot be used as a reference"):
        nodes._resolve_runtime(
            object(), _branch_maps(), profile="reference", projection_precision=precision
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
    assert nodes._auto_execution_mode(model, "hybrid") == "serial"

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=96 * gib),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (70 * gib, 96 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib, int(4.5 * gib)) == "resident"
    assert nodes._auto_tile_frames(model, "resident") == 0
    assert nodes._auto_execution_mode(model, "resident") == "parallel"

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (5 * gib, 96 * gib))
    assert nodes._auto_branch_mode(model, 5 * gib, int(4.5 * gib)) == "stream"
    assert nodes._auto_execution_mode(model, "stream") == "serial"


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
    assert snapshot["adapters"]["active"] == ["default", "turbo"]
    assert snapshot["adapters"]["lora_mode"] == "bypass"
    assert snapshot["cuda"]["available"] is False

    node = KireiVDNH3RuntimeReport()
    sentinel = object()
    response = node.report(model, after=sentinel)
    passthrough, text = response["result"]
    assert passthrough is model
    assert response["ui"]["text"] == [text]
    parsed = json.loads(text)
    assert parsed["attention_last"] == "grouped"
    assert parsed["diagnostics"]["attention.softmax"]["last_ms"] == 1.25
    schema = node.INPUT_TYPES()
    assert schema["optional"]["after"][0] == "*"
    assert node.OUTPUT_NODE is True


def test_auto_bf16_base_merges_adapters_exactly(monkeypatch):
    monkeypatch.setattr(nodes, "detect_base_precision", lambda model: "bf16")
    monkeypatch.setattr(nodes, "_auto_branch_mode", lambda *args: "resident")
    monkeypatch.setattr(nodes, "_auto_execution_mode", lambda *args: "serial")
    monkeypatch.setattr(nodes, "_auto_tile_frames", lambda *args: 0)
    runtime = nodes._resolve_runtime(object(), _branch_maps(), profile="auto")
    # One fp32 delta folded into BF16 weights is exact and removes the runtime GEMMs.
    assert runtime["lora_mode"] == "merge"
    assert runtime["projection_precision"] == "bf16"
    low = nodes._resolve_runtime(object(), _branch_maps(), profile="low_vram")
    assert low["lora_mode"] == "bypass"


def test_auto_quantized_bases_keep_the_exact_bypass(monkeypatch):
    monkeypatch.setattr(nodes, "_auto_branch_mode", lambda *args: "resident")
    monkeypatch.setattr(nodes, "_auto_execution_mode", lambda *args: "serial")
    monkeypatch.setattr(nodes, "_auto_tile_frames", lambda *args: 0)
    for precision in ("int8", "fp8", "nvfp4", "mxfp8"):
        monkeypatch.setattr(nodes, "detect_base_precision", lambda model, p=precision: p)
        runtime = nodes._resolve_runtime(object(), _branch_maps(), profile="auto")
        assert runtime["base_precision"] == precision
        assert runtime["lora_mode"] == "bypass"
        assert runtime["projection_precision"] == "bf16"


def test_auto_branch_mode_budgets_against_the_base_model_size(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nodes, "_model_cuda_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=96 * gib),
    )
    # Before ComfyUI loads the base the card looks almost empty.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (94 * gib, 96 * gib))
    int8_base = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 34 * gib)
    bf16_base = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 66 * gib)
    huge_base = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 82 * gib)
    assert nodes._auto_branch_mode(int8_base, 5 * gib, int(4.5 * gib)) == "resident"
    assert nodes._auto_branch_mode(bf16_base, 5 * gib, int(4.5 * gib)) == "resident"
    assert nodes._auto_branch_mode(huge_base, 5 * gib, int(4.5 * gib)) == "hybrid"


def test_layout_snapshot_reports_global_rows():
    from vdn_h3.benchmark import _layout_snapshot
    from vdn_h3.layout import VDNLayout

    layout = VDNLayout(
        seq_len=1500, video_start=1200, video_end=1500, num_frames=3, tokens_per_frame=100,
        frame_size=(10, 10), text_start=0, text_len=200, bounds=((0, 2),) * 3,
        full_cover=True, anchor_frames="both",
    )
    snapshot = _layout_snapshot(SimpleNamespace(last_layout=layout))
    assert snapshot["video_rows"] == 300
    assert snapshot["text_rows"] == 200
    assert snapshot["other_global_rows"] == 1000
    assert snapshot["global_rows"] == 1200
    assert abs(snapshot["global_fraction"] - 0.8) < 1e-9
