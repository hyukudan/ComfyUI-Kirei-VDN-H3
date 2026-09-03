from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "benchmarks" / "scenarios.json"
COMPARE = ROOT / "benchmarks" / "compare_results.py"
RECORD = ROOT / "benchmarks" / "record_result.py"


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload():
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))


def test_recipe_table_matches_released_intent():
    payload = _payload()
    recipes = payload["recipes"]
    turbo = recipes["turbo_conventional_4"]
    dmd = recipes["vdn_stage_dmd_8"]
    stage_b = recipes["vdn_stage_b_50"]

    assert turbo["steps"] == 4
    assert dmd["steps"] == 8
    assert stage_b["steps"] == 50
    for recipe in (turbo, dmd, stage_b):
        assert recipe["video_shift"] == 12.0
        assert recipe["audio_shift"] == 3.0
    assert dmd["required_adapters"] == ["default", "turbo"]
    assert stage_b["required_adapters"] == ["default"]


def test_active_scenarios_follow_their_own_recipe():
    payload = _payload()
    recipes = payload["recipes"]
    for scenario in payload["scenarios"]:
        if not scenario.get("active"):
            continue
        recipe = recipes[scenario["recipe_id"]]
        assert scenario["steps"] == recipe["steps"], scenario["id"]


def test_product_groups_compare_same_output_objective_not_same_nfe():
    payload = _payload()
    recipes = payload["recipes"]
    groups = {}
    for scenario in payload["scenarios"]:
        group = scenario.get("product_group")
        if not scenario.get("active") or not group:
            continue
        recipe = recipes[scenario["recipe_id"]]
        objective = (
            scenario["width"],
            scenario["height"],
            scenario["frames"],
            scenario["quality_target"],
            recipe["scheduler_family"],
            recipe["video_shift"],
            recipe["audio_shift"],
        )
        groups.setdefault(group, {"objectives": set(), "steps": set()})
        groups[group]["objectives"].add(objective)
        groups[group]["steps"].add(scenario["steps"])
    assert groups
    assert all(len(item["objectives"]) == 1 for item in groups.values())
    # At least one real product group must compare recipe-faithful Turbo 4 NFE to VDN 8 NFE.
    assert any(item["steps"] == {4, 8} for item in groups.values())


def test_technical_groups_remain_strict_same_nfe_and_recipe():
    payload = _payload()
    groups = {}
    for scenario in payload["scenarios"]:
        group = scenario.get("technical_group")
        if not scenario.get("active") or not group:
            continue
        value = (scenario["width"], scenario["height"], scenario["frames"], scenario["steps"], scenario["recipe_id"])
        groups.setdefault(group, set()).add(value)
    assert groups
    assert all(len(values) == 1 for values in groups.values())


def test_release_fidelity_and_product_geometries_exist():
    payload = _payload()
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    stage_b = scenarios["vdn_stage_b_bf16_50step_1344x768_345"]
    turbo = scenarios["turbo4_1344x768_345"]
    dmd = scenarios["vdn_dmd_bf16_8step_1344x768_345"]

    assert (stage_b["width"], stage_b["height"], stage_b["frames"], stage_b["steps"]) == (1344, 768, 345, 50)
    assert stage_b["apply_turbo_adapter"] is False
    assert turbo["steps"] == 4
    assert dmd["steps"] == 8
    assert turbo["product_group"] == dmd["product_group"]

    active = [item for item in payload["scenarios"] if item.get("active")]
    assert any(item["frames"] == 241 for item in active)
    assert any(item["frames"] == 401 for item in active)


def _row(**overrides):
    value = {
        "scenario_id": "vdn_a",
        "product_group": "product",
        "technical_group": "technical",
        "quality_target": "fewstep_production_quality",
        "recipe_id": "vdn_stage_dmd_8",
        "quality_status": "qualified",
        "quality_gate_required": True,
        "comparable": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": 8,
        "seed": 1,
        "scheduler": "euler",
        "scheduler_family": "minimax_h3_flow_euler_v12_a3",
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "prompt_hash": "p",
        "sampler_seconds": 10.0,
    }
    value.update(overrides)
    return value


def test_product_comparator_allows_recipe_faithful_mixed_nfe():
    module = _module(COMPARE, "vdn_bench_compare")
    rows = [
        _row(
            scenario_id="turbo",
            technical_group=None,
            recipe_id="turbo_conventional_4",
            steps=4,
            sampler_seconds=8.0,
        ),
        _row(scenario_id="vdn", sampler_seconds=9.0),
    ]
    out = module.product_comparisons(module.summarize(rows))["product"]
    assert [item["steps"] for item in out["ranking"]] == [4, 8]
    assert out["speed_claim_eligible"] is True


def test_technical_comparator_rejects_mixed_steps_or_recipe():
    module = _module(COMPARE, "vdn_bench_compare_technical")
    rows = [
        _row(scenario_id="a"),
        _row(scenario_id="b", steps=4, recipe_id="turbo_conventional_4", sampler_seconds=9.0),
    ]
    summary = module.summarize(rows)
    try:
        module.technical_comparisons(summary)
    except ValueError as exc:
        assert "same_nfe_technical" in str(exc)
    else:
        raise AssertionError("technical comparison must reject mixed NFE/recipes")


def test_quality_gate_controls_product_speed_claim():
    module = _module(COMPARE, "vdn_bench_compare_quality")
    pending_rows = [
        _row(
            scenario_id="turbo",
            technical_group=None,
            recipe_id="turbo_conventional_4",
            steps=4,
            sampler_seconds=8.0,
        ),
        _row(scenario_id="vdn", sampler_seconds=9.0, quality_status="pending"),
    ]
    pending = module.product_comparisons(module.summarize(pending_rows))["product"]
    assert pending["speed_claim_eligible"] is False

    pending_rows[1]["quality_status"] = "qualified"
    qualified = module.product_comparisons(module.summarize(pending_rows))["product"]
    assert qualified["speed_claim_eligible"] is True


def test_record_result_validates_stage_dmd_steps_and_shifts():
    module = _module(RECORD, "vdn_bench_record")
    payload = _payload()
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    scenario = scenarios["vdn_dmd_bf16_8step_608x352_121"]
    recipe = payload["recipes"][scenario["recipe_id"]]
    measurement = {
        "scenario_id": scenario["id"],
        "sampler_seconds": 1.0,
        "sampling": {"video_shift": 12.0, "audio_shift": 3.0},
        "runtime_report": {"checkpoint_recipe": {"turbo_num_steps": 8}},
    }
    row = module.build_result(
        measurement,
        scenario,
        recipe,
        seed=1,
        scheduler="euler",
        prompt_hash="p",
        quality_status="pending",
    )
    assert row["steps"] == 8
    assert row["recipe_id"] == "vdn_stage_dmd_8"

    bad = dict(measurement)
    bad["sampling"] = {"video_shift": 7.0, "audio_shift": 3.0}
    try:
        module.build_result(
            bad,
            scenario,
            recipe,
            seed=1,
            scheduler="euler",
            prompt_hash="p",
            quality_status="pending",
        )
    except ValueError as exc:
        assert "sampling shifts" in str(exc)
    else:
        raise AssertionError("wrong H3 shift must be rejected")
