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


def test_canonical_recipes_lock_sampler_scheduler_and_denoise():
    payload = _payload()
    recipes = payload["recipes"]
    larry = recipes["larry_turbo_v4_quality8"]
    dmd = recipes["vdn_stage_dmd_8"]
    stage_b = recipes["vdn_stage_b_50"]
    native = recipes["native_standard_20"]

    assert (larry["steps"], larry["sampler_name"], larry["scheduler_name"], larry["denoise"]) == (8, "euler", "simple", 1.0)
    assert larry["lora_strength"] == 1.0
    assert (dmd["steps"], dmd["sampler_name"], dmd["scheduler_name"], dmd["denoise"]) == (8, "euler", "simple", 1.0)
    assert (stage_b["steps"], stage_b["sampler_name"], stage_b["scheduler_name"], stage_b["denoise"]) == (50, "euler", "simple", 1.0)
    assert native["sampler_name"] == "res_multistep"
    assert native["scheduler_name"] == "simple"

    for recipe in (larry, dmd, stage_b, native):
        assert recipe["video_shift"] == 12.0
        assert recipe["audio_shift"] == 3.0

    assert dmd["required_adapters"] == ["default", "turbo"]
    assert stage_b["required_adapters"] == ["default"]


def test_no_active_vdn_scenario_can_use_res_multistep():
    payload = _payload()
    recipes = payload["recipes"]
    active_vdn = [
        item
        for item in payload["scenarios"]
        if item.get("active") and recipes[item["recipe_id"]].get("expects_vdn_runtime")
    ]
    assert active_vdn
    for scenario in active_vdn:
        recipe = recipes[scenario["recipe_id"]]
        assert recipe["sampler_name"] == "euler", scenario["id"]
        assert recipe["scheduler_name"] == "simple", scenario["id"]
        assert recipe["denoise"] == 1.0, scenario["id"]
        assert recipe["sampler_name"] != "res_multistep", scenario["id"]


def test_active_scenarios_follow_their_recipe_steps():
    payload = _payload()
    recipes = payload["recipes"]
    for scenario in payload["scenarios"]:
        if scenario.get("active"):
            assert scenario["steps"] == recipes[scenario["recipe_id"]]["steps"], scenario["id"]


def test_primary_product_groups_share_exact_quality_trajectory():
    payload = _payload()
    recipes = payload["recipes"]
    groups = {}
    for scenario in payload["scenarios"]:
        group = scenario.get("product_group")
        if not scenario.get("active") or not group:
            continue
        recipe = recipes[scenario["recipe_id"]]
        objective = (
            scenario["width"], scenario["height"], scenario["frames"], scenario["steps"],
            recipe["sampler_name"], recipe["scheduler_name"], recipe["denoise"],
            recipe["video_shift"], recipe["audio_shift"], scenario["quality_target"],
        )
        groups.setdefault(group, set()).add(objective)
    assert groups
    assert all(len(values) == 1 for values in groups.values())
    # The clean Larry control and VDN-DMD now compare on the same 8-NFE Euler/simple path.
    assert all(next(iter(values))[3:7] == (8, "euler", "simple", 1.0) for values in groups.values())


def test_release_fidelity_and_long_geometries_exist():
    scenarios = {item["id"]: item for item in _payload()["scenarios"]}
    stage_b = scenarios["vdn_stage_b_bf16_50step_1344x768_345"]
    larry = scenarios["larry8_1344x768_345"]
    dmd = scenarios["vdn_dmd_bf16_8step_1344x768_345"]
    assert (stage_b["width"], stage_b["height"], stage_b["frames"], stage_b["steps"]) == (1344, 768, 345, 50)
    assert stage_b["apply_turbo_adapter"] is False
    assert larry["steps"] == dmd["steps"] == 8
    assert larry["product_group"] == dmd["product_group"]

    active = [item for item in scenarios.values() if item.get("active")]
    assert any(item["frames"] == 241 for item in active)
    assert any(item["frames"] == 401 for item in active)


def _row(**overrides):
    value = {
        "scenario_id": "vdn_a",
        "product_group": "quality8",
        "technical_group": "quality8",
        "quality_target": "fewstep_quality_8nfe",
        "recipe_id": "vdn_stage_dmd_8",
        "quality_status": "qualified",
        "quality_gate_required": True,
        "comparable": True,
        "width": 608,
        "height": 352,
        "frames": 241,
        "steps": 8,
        "sampler_name": "euler",
        "scheduler_name": "simple",
        "denoise": 1.0,
        "seed": 1,
        "scheduler_family": "minimax_h3_flow_v12_a3",
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "prompt_hash": "p",
        "sampler_seconds": 10.0,
    }
    value.update(overrides)
    return value


def test_product_comparator_accepts_larry_and_vdn_on_same_clean_trajectory():
    module = _module(COMPARE, "vdn_bench_compare")
    rows = [
        _row(scenario_id="larry", recipe_id="larry_turbo_v4_quality8", sampler_seconds=8.0),
        _row(scenario_id="vdn", sampler_seconds=9.0),
    ]
    out = module.product_comparisons(module.summarize(rows))["quality8"]
    assert out["trajectory"] == {
        "steps": 8,
        "sampler_name": "euler",
        "scheduler_name": "simple",
        "denoise": 1.0,
    }
    assert out["ranking"][0]["scenario_id"] == "larry"


def test_product_comparator_rejects_res_multistep_or_beta_mismatch():
    module = _module(COMPARE, "vdn_bench_compare_bad_sampler")
    rows = [
        _row(scenario_id="larry", recipe_id="larry_turbo_v4_quality8"),
        _row(scenario_id="vdn", sampler_name="res_multistep", sampler_seconds=9.0),
    ]
    summary = module.summarize(rows)
    try:
        module.product_comparisons(summary)
    except ValueError as exc:
        assert "sampler/scheduler" in str(exc)
    else:
        raise AssertionError("different samplers must never be ranked as comparable")


def test_technical_comparator_partitions_by_recipe_and_compares_vdn_variants():
    module = _module(COMPARE, "vdn_bench_compare_technical")
    rows = [
        _row(scenario_id="larry", recipe_id="larry_turbo_v4_quality8", sampler_seconds=8.0),
        _row(scenario_id="vdn_bf16", sampler_seconds=10.0),
        _row(scenario_id="vdn_int8", sampler_seconds=9.0),
    ]
    out = module.technical_comparisons(module.summarize(rows))
    key = "quality8::vdn_stage_dmd_8"
    assert key in out
    assert [item["scenario_id"] for item in out[key]["ranking"]] == ["vdn_int8", "vdn_bf16"]
    assert all("larry" not in item["scenario_id"] for item in out[key]["ranking"])


def test_quality_gate_controls_product_speed_claim():
    module = _module(COMPARE, "vdn_bench_compare_quality")
    rows = [
        _row(scenario_id="larry", recipe_id="larry_turbo_v4_quality8", sampler_seconds=8.0),
        _row(scenario_id="vdn", sampler_seconds=9.0, quality_status="pending"),
    ]
    assert module.product_comparisons(module.summarize(rows))["quality8"]["speed_claim_eligible"] is False
    rows[1]["quality_status"] = "qualified"
    assert module.product_comparisons(module.summarize(rows))["quality8"]["speed_claim_eligible"] is True


def test_record_result_rejects_res_multistep_for_stage_dmd():
    module = _module(RECORD, "vdn_bench_record")
    payload = _payload()
    scenario = next(item for item in payload["scenarios"] if item["id"] == "vdn_dmd_bf16_8step_608x352_121")
    recipe = payload["recipes"][scenario["recipe_id"]]
    measurement = {
        "scenario_id": scenario["id"],
        "sampler_seconds": 1.0,
        "sampling_plan": {
            "verified": True,
            "scenario_id": scenario["id"],
            "recipe_id": scenario["recipe_id"],
            "sampler_name": "res_multistep",
            "scheduler_name": "simple",
            "steps": 8,
            "denoise": 1.0,
            "video_shift": 12.0,
            "audio_shift": 3.0,
        },
        "sampling": {"video_shift": 12.0, "audio_shift": 3.0},
        "runtime_report": {
            "profile": "auto",
            "projection": {"precision": "bf16"},
            "checkpoint_recipe": {"turbo_num_steps": 8},
        },
    }
    try:
        module.build_result(
            measurement, scenario, recipe, seed=1, prompt_hash="p", quality_status="pending"
        )
    except ValueError as exc:
        assert "sampler_name" in str(exc) or "res_multistep" in str(exc)
    else:
        raise AssertionError("Stage-DMD res_multistep measurement must be rejected")
