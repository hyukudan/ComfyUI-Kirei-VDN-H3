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
        "quality_target": "fewstep_quality_8nfe",
        "recipe_id": "vdn_stage_dmd_8",
        "quality_gate_required": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": steps,
        "model_variant": "vdn_auto",
        "profile": "auto",
        "projection_precision": precision,
    }


def _vdn_recipe():
    return {
        "label": "OpenVDN Stage-DMD release",
        "steps": 8,
        "sampler_name": "euler",
        "scheduler_name": "simple",
        "denoise": 1.0,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "scheduler_family": "minimax_h3_flow_v12_a3",
        "expects_vdn_runtime": True,
        "runtime_turbo_num_steps": 8,
        "required_adapters": ["default", "turbo"],
        "required_adapter_strength": 1.0,
    }


def _plan(**overrides):
    value = {
        "verified": True,
        "scenario_id": "vdn",
        "recipe_id": "vdn_stage_dmd_8",
        "sampler_name": "euler",
        "scheduler_name": "simple",
        "steps": 8,
        "denoise": 1.0,
        "video_shift": 12.0,
        "audio_shift": 3.0,
    }
    value.update(overrides)
    return value


def _measurement(precision="bf16", profile="auto", plan=None):
    return {
        "scenario_id": "vdn",
        "run_kind": "warm",
        "sampler_seconds": 10.0,
        "peak_vram_bytes": 123,
        "sampling_plan": _plan() if plan is None else plan,
        "sampling": {"video_shift": 12.0, "audio_shift": 3.0},
        "runtime_report": {
            "profile": profile,
            "projection": {"precision": precision},
            "checkpoint_recipe": {"turbo_num_steps": 8},
        },
    }


def test_record_result_accepts_canonical_stage_dmd_trajectory():
    module = _module()
    row = module.build_result(
        _measurement(),
        _vdn_scenario(8),
        _vdn_recipe(),
        seed=1,
        prompt_hash="p",
        quality_status="pending",
    )
    assert row["steps"] == 8
    assert row["sampler_name"] == "euler"
    assert row["scheduler_name"] == "simple"
    assert row["denoise"] == 1.0
    assert row["recipe_id"] == "vdn_stage_dmd_8"


def test_record_result_rejects_unverified_sampling_plan():
    module = _module()
    measurement = _measurement(plan={"verified": False})
    with pytest.raises(ValueError, match="verified Kirei Benchmark Sampling"):
        module.build_result(
            measurement,
            _vdn_scenario(),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_res_multistep_on_stage_dmd():
    module = _module()
    measurement = _measurement(plan=_plan(sampler_name="res_multistep"))
    with pytest.raises(ValueError, match="sampling plan mismatch|res_multistep"):
        module.build_result(
            measurement,
            _vdn_scenario(),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_beta_scheduler_on_stage_dmd():
    module = _module()
    measurement = _measurement(plan=_plan(scheduler_name="beta"))
    with pytest.raises(ValueError, match="scheduler_name"):
        module.build_result(
            measurement,
            _vdn_scenario(),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_wrong_denoise_or_step_count():
    module = _module()
    with pytest.raises(ValueError, match="denoise"):
        module.build_result(
            _measurement(plan=_plan(denoise=0.8)),
            _vdn_scenario(),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )
    with pytest.raises(ValueError, match="steps"):
        module.build_result(
            _measurement(plan=_plan(steps=6)),
            _vdn_scenario(),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_wrong_model_sampling_shift():
    module = _module()
    measurement = _measurement()
    measurement["sampling"] = {"video_shift": 6.0, "audio_shift": 3.0}
    with pytest.raises(ValueError, match="model sampling shifts"):
        module.build_result(
            measurement,
            _vdn_scenario(),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_precision_fallback_under_wrong_label():
    module = _module()
    with pytest.raises(ValueError, match="projection_precision='int8'"):
        module.build_result(
            _measurement(precision="bf16"),
            _vdn_scenario(precision="int8"),
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_record_result_rejects_wrong_profile_label():
    module = _module()
    scenario = _vdn_scenario()
    scenario["model_variant"] = "vdn_max_speed"
    scenario["profile"] = "max_speed"
    with pytest.raises(ValueError, match="profile='max_speed'"):
        module.build_result(
            _measurement(profile="auto"),
            scenario,
            _vdn_recipe(),
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )


def test_non_vdn_control_rejects_vdn_runtime_state():
    module = _module()
    scenario = {
        "id": "larry",
        "active": True,
        "comparable": True,
        "product_group": "product",
        "quality_target": "fewstep_quality_8nfe",
        "recipe_id": "larry_turbo_v4_quality8",
        "quality_gate_required": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": 8,
    }
    recipe = {
        "label": "Larry",
        "steps": 8,
        "sampler_name": "euler",
        "scheduler_name": "simple",
        "denoise": 1.0,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "scheduler_family": "minimax_h3_flow_v12_a3",
        "expects_vdn_runtime": False,
    }
    measurement = {
        "scenario_id": "larry",
        "sampler_seconds": 1.0,
        "sampling_plan": {
            "verified": True,
            "scenario_id": "larry",
            "recipe_id": "larry_turbo_v4_quality8",
            "sampler_name": "euler",
            "scheduler_name": "simple",
            "steps": 8,
            "denoise": 1.0,
            "video_shift": 12.0,
            "audio_shift": 3.0,
        },
        "sampling": {"video_shift": 12.0, "audio_shift": 3.0},
        "runtime_report": {"checkpoint_recipe": {"turbo_num_steps": 8}},
    }
    with pytest.raises(ValueError, match="non-VDN control"):
        module.build_result(
            measurement,
            scenario,
            recipe,
            seed=1,
            prompt_hash="p",
            quality_status="pending",
        )
