from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "benchmarks" / "scenarios.json"
COMPARE = ROOT / "benchmarks" / "compare_results.py"


def _module():
    spec = importlib.util.spec_from_file_location("vdn_bench_compare", COMPARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_comparable_groups_keep_one_generation_objective():
    payload = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    groups = {}
    for scenario in payload["scenarios"]:
        if not scenario.get("comparable", True):
            continue
        key = scenario["comparison_group"]
        objective = (
            scenario["width"],
            scenario["height"],
            scenario["frames"],
            scenario["steps"],
        )
        groups.setdefault(key, set()).add(objective)
    assert groups
    assert all(len(objectives) == 1 for objectives in groups.values())


def test_native20_is_not_ranked_with_distilled4():
    payload = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    native = scenarios["native20_608x352_121"]
    turbo = scenarios["turbo4_608x352_121"]
    assert native["steps"] == 20
    assert turbo["steps"] == 4
    assert native["comparison_group"] != turbo["comparison_group"]


def test_long_video_primary_and_stress_geometries_exist():
    payload = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    frames = {item["frames"] for item in payload["scenarios"] if item.get("comparable", True)}
    assert 241 in frames
    assert 401 in frames


def test_result_comparator_rejects_mixed_objectives():
    module = _module()
    rows = [
        {
            "scenario_id": "a",
            "comparison_group": "same",
            "width": 608,
            "height": 352,
            "frames": 241,
            "steps": 4,
            "seed": 1,
            "scheduler": "s",
            "sampler_seconds": 10.0,
        },
        {
            "scenario_id": "b",
            "comparison_group": "same",
            "width": 608,
            "height": 352,
            "frames": 241,
            "steps": 20,
            "seed": 1,
            "scheduler": "s",
            "sampler_seconds": 9.0,
        },
    ]
    summary = module.summarize(rows)
    try:
        module.comparisons(summary)
    except ValueError as exc:
        assert "mixes resolution/frames/steps/seed/scheduler" in str(exc)
    else:
        raise AssertionError("mixed 4-step/20-step results must not be ranked together")
