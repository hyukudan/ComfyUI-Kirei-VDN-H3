from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "record_result.py"


def _module():
    spec = importlib.util.spec_from_file_location("vdn_bench_record", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _vdn_scenario(steps=8, precision="bf16"):
    return {
        "id": "vdn",
        "active": True,
        "comparable": True,
        "product_group": "product",
        "technical_group": "technical",
        "quality_target": "fewstep_production_quality",
        "recipe_id": "vdn_stage_dmd_8",
        "quality_gate_required": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": steps,
        "model_variant": "vdn_auto",
        "projection_precision": precision,
    }


def _vdn_recipe():
    return {
        "label": "VDN-H3 Stage-DMD release",
        "steps": 8,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "scheduler_family": "minimax_h3_flow_euler_v12_a3",
        "expects_vdn_runtime": True,
        "runtime_turbo_num_steps": 8,
        "required_adapters": ["default", "turbo"],
        "required_adapter_strength": 1.0,
    }


def _measurement(precision="bf16", profile="auto"):
    return {
        "scenario_id": "vdn",
        "run_kind": "warm",
        "sampler_seconds": 10.0,
        "peak_vram_bytes": 123,
        "sampling": {"video_shift": 12.0, "audio_shift": 3.0},
        "runtime_report": {
            "profile": profile,
            "projection": {"precision": precision},
            "checkpoint_recipe": {"turbo_num_steps": 8},
        },
    }


def test_record_result_accepts_matching_stage_dmd_recipe():
    module = _module()
    row = module.build_result(
        _measurement(),
        _vdn_scenario(8),
        _vdn_recipe(),
        seed=1,
        scheduler="euler",
        prompt_hash="p",
        quality_status="pending",
    )
    assert row["steps"] == 8
    assert row["recipe_id"] == "vdn_stage_dmd_8"
    assert row["video_shift"] == 12.0
    assert row["quality_status"] == "pending"


def test_record_result_rejects_active_stage_dmd_at_four_steps():
    module = _module()
    with pytest.raises(ValueError, match="requires 8"):
        module.build_result(
            _measurement(),
            _vdn_scenario(4),
            _vdn_recipe(),
            seed=1,
            scheduler="euler",
            prompt_hash="p",
            quality_status="diagnostic",
        )


def test_record_result_rejects_wrong_sampling_shift():
    module = _module()
    measurement = _measurement()
    measurement["sampling"] = {"video_shift": 6.0, "audio_shift": 3.0}
    with pytest.raises(ValueError, match="sampling shifts"):
        module.build_result(
            measurement,
            _vdn_scenario(8),
            _vdn_recipe(),
            seed=1,
            scheduler="euler",
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_precision_fallback_under_wrong_label():
    module = _module()
    with pytest.raises(ValueError, match="projection_precision='int8'"):
        module.build_result(
            _measurement(precision="bf16"),
            _vdn_scenario(8, precision="int8"),
            _vdn_recipe(),
            seed=1,
            scheduler="euler",
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_wrong_profile_label():
    module = _module()
    scenario = _vdn_scenario(8)
    scenario["model_variant"] = "vdn_max_speed"
    with pytest.raises(ValueError, match="profile='max_speed'"):
        module.build_result(
            _measurement(profile="auto"),
            scenario,
            _vdn_recipe(),
            seed=1,
            scheduler="euler",
            prompt_hash="p",
            quality_status="pending",
        )


def test_non_vdn_control_rejects_vdn_runtime_state():
    module = _module()
    scenario = {
        "id": "turbo",
        "active": True,
        "comparable": True,
        "product_group": "product",
        "quality_target": "fewstep_production_quality",
        "recipe_id": "turbo_conventional_4",
        "quality_gate_required": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": 4,
    }
    recipe = {
        "label": "Turbo",
        "steps": 4,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "scheduler_family": "minimax_h3_flow_euler_v12_a3",
        "expects_vdn_runtime": False,
    }
    measurement = {
        "scenario_id": "turbo",
        "sampler_seconds": 1.0,
        "sampling": {"video_shift": 12.0, "audio_shift": 3.0},
        "runtime_report": {"checkpoint_recipe": {"turbo_num_steps": 8}},
    }
    with pytest.raises(ValueError, match="non-VDN control"):
        module.build_result(
            measurement,
            scenario,
            recipe,
            seed=1,
            scheduler="euler",
            prompt_hash="p",
            quality_status="pending",
        )
