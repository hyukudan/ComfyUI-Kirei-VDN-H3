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


def _scenario(steps=8):
    return {
        "id": "vdn",
        "comparison_group": "g",
        "active": True,
        "comparable": True,
        "quality_target": "checkpoint_declared_distilled8",
        "recipe": "turbo_num_steps_8_same_prompt_seed_scheduler",
        "quality_gate_required": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": steps,
    }


def _measurement():
    return {
        "scenario_id": "vdn",
        "run_kind": "warm",
        "sampler_seconds": 10.0,
        "peak_vram_bytes": 123,
        "runtime_report": {
            "checkpoint_recipe": {"turbo_num_steps": 8},
        },
    }


def test_record_result_accepts_matching_checkpoint_recipe():
    module = _module()
    row = module.build_result(
        _measurement(),
        _scenario(8),
        {"checkpoint_declared_turbo_steps": 8},
        seed=1,
        scheduler="s",
        prompt_hash="p",
        quality_status="pending",
    )
    assert row["steps"] == 8
    assert row["checkpoint_turbo_num_steps"] == 8
    assert row["quality_status"] == "pending"


def test_record_result_rejects_runtime_step_mismatch():
    module = _module()
    with pytest.raises(ValueError, match="turbo_num_steps=8"):
        module.build_result(
            _measurement(),
            _scenario(4),
            {"checkpoint_declared_turbo_steps": 8},
            seed=1,
            scheduler="s",
            prompt_hash="p",
            quality_status="diagnostic",
        )
