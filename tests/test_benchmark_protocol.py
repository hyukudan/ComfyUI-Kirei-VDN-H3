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


def _payload():
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))


def test_comparable_groups_keep_one_generation_objective():
    payload = _payload()
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
            scenario["quality_target"],
            scenario["recipe"],
        )
        groups.setdefault(key, set()).add(objective)
    assert groups
    assert all(len(objectives) == 1 for objectives in groups.values())


def test_active_distilled_checkpoint_recipe_is_eight_steps():
    payload = _payload()
    assert payload["rules"]["checkpoint_declared_turbo_steps"] == 8
    active_distilled = [
        item
        for item in payload["scenarios"]
        if item.get("active")
        and item.get("comparable")
        and item["quality_target"].startswith("checkpoint_declared_distilled8")
    ]
    assert active_distilled
    assert all(item["steps"] == 8 for item in active_distilled)
    assert all("turbo_num_steps_8" in item["recipe"] for item in active_distilled)


def test_historical_four_step_paths_are_not_active_comparisons():
    payload = _payload()
    old = [
        item
        for item in payload["scenarios"]
        if item["steps"] == 4
    ]
    assert old
    assert all(not item.get("active", False) for item in old)
    assert all(not item.get("comparable", True) for item in old)


def test_native20_is_not_ranked_with_distilled8():
    payload = _payload()
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    native = scenarios["native20_608x352_121"]
    turbo = scenarios["turbo8_608x352_121"]
    assert native["steps"] == 20
    assert turbo["steps"] == 8
    assert native["quality_target"] != turbo["quality_target"]
    assert native["comparison_group"] != turbo["comparison_group"]


def test_quality_and_long_video_geometries_exist():
    payload = _payload()
    active = [item for item in payload["scenarios"] if item.get("active")]
    assert any(
        item["width"] == 960 and item["height"] == 544 and item["frames"] == 121 and item["steps"] == 8
        for item in active
    )
    assert any(item["frames"] == 241 and item["steps"] == 8 for item in active)
    assert any(item["frames"] == 401 and item["steps"] == 8 for item in active)


def _row(**overrides):
    value = {
        "scenario_id": "a",
        "comparison_group": "same",
        "quality_target": "checkpoint_declared_distilled8",
        "recipe": "turbo_num_steps_8_same_prompt_seed_scheduler",
        "quality_status": "qualified",
        "quality_gate_required": True,
        "comparable": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": 8,
        "seed": 1,
        "scheduler": "s",
        "prompt_hash": "p",
        "sampler_seconds": 10.0,
    }
    value.update(overrides)
    return value


def test_result_comparator_rejects_mixed_recipes_or_objectives():
    module = _module()
    rows = [
        _row(scenario_id="a"),
        _row(
            scenario_id="b",
            recipe="wrong_4step_recipe",
            sampler_seconds=9.0,
        ),
    ]
    summary = module.summarize(rows)
    try:
        module.comparisons(summary)
    except ValueError as exc:
        assert "prompt/quality_target/recipe" in str(exc)
    else:
        raise AssertionError("mixed recipes must not be ranked together")


def test_result_comparator_rejects_mixed_steps():
    module = _module()
    rows = [
        _row(scenario_id="a"),
        _row(scenario_id="b", steps=4, sampler_seconds=9.0),
    ]
    summary = module.summarize(rows)
    try:
        module.comparisons(summary)
    except ValueError as exc:
        assert "resolution/frames/steps" in str(exc)
    else:
        raise AssertionError("mixed 8-step/4-step results must not be ranked together")


def test_quality_gate_controls_same_quality_speed_claim():
    module = _module()
    pending = module.comparisons(
        module.summarize(
            [
                _row(scenario_id="turbo", sampler_seconds=10.0),
                _row(
                    scenario_id="vdn",
                    sampler_seconds=9.0,
                    quality_status="pending",
                ),
            ]
        )
    )["same"]
    assert pending["speed_claim_eligible"] is False

    qualified = module.comparisons(
        module.summarize(
            [
                _row(scenario_id="turbo", sampler_seconds=10.0),
                _row(
                    scenario_id="vdn",
                    sampler_seconds=9.0,
                    quality_status="qualified",
                ),
            ]
        )
    )["same"]
    assert qualified["speed_claim_eligible"] is True
    assert qualified["ranking"][0]["scenario_id"] == "vdn"
    assert qualified["ranking"][0]["same_quality_speed_claim"] is True
