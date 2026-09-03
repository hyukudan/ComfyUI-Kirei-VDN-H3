from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

QUALITY_STATUSES = {"pending", "qualified", "failed", "diagnostic"}


def load_scenarios(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    recipes = dict(payload.get("recipes", {}))
    return payload, scenarios, recipes


def _close(a, b, tol=1e-6):
    return a is not None and b is not None and math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0.0)


def _runtime_adapters(runtime):
    if not isinstance(runtime, dict):
        return None
    adapters = runtime.get("adapters")
    if not isinstance(adapters, dict):
        return None
    active = adapters.get("active")
    strengths = adapters.get("strengths")
    return {
        "active": list(active) if isinstance(active, (list, tuple)) else None,
        "strengths": dict(strengths) if isinstance(strengths, dict) else {},
    }


def _expected_vdn_runtime(scenario: dict) -> dict:
    variant = str(scenario.get("model_variant", ""))
    expected = {}
    if variant == "vdn_max_speed":
        expected["profile"] = "max_speed"
    elif variant == "vdn_stage_b_reference":
        expected["profile"] = "reference"
    elif variant.startswith("vdn_"):
        expected["profile"] = str(scenario.get("profile", "auto"))
    precision = scenario.get("projection_precision")
    if precision:
        expected["projection_precision"] = str(precision)
    return expected


def _validate_vdn_runtime_label(runtime: dict, scenario: dict):
    expected = _expected_vdn_runtime(scenario)
    profile = expected.get("profile")
    if profile is not None and runtime.get("profile") != profile:
        raise ValueError(
            f"scenario {scenario['id']!r} expects VDN profile={profile!r}, but Runtime Report "
            f"resolved profile={runtime.get('profile')!r}"
        )
    precision = expected.get("projection_precision")
    if precision is not None:
        projection = runtime.get("projection")
        got = projection.get("precision") if isinstance(projection, dict) else None
        if got != precision:
            raise ValueError(
                f"scenario {scenario['id']!r} expects projection_precision={precision!r}, "
                f"but Runtime Report resolved {got!r}; do not record a fallback under the wrong label"
            )


def _expected_sampling_plan(scenario: dict, recipe: dict) -> dict:
    return {
        "scenario_id": str(scenario["id"]),
        "recipe_id": str(scenario["recipe_id"]),
        "sampler_name": str(recipe["sampler_name"]),
        "scheduler_name": str(recipe["scheduler_name"]),
        "steps": int(recipe["steps"]),
        "denoise": float(recipe.get("denoise", 1.0)),
        "video_shift": float(recipe["video_shift"]),
        "audio_shift": float(recipe["audio_shift"]),
    }


def _validate_sampling_plan(measurement: dict, scenario: dict, recipe: dict) -> dict:
    plan = measurement.get("sampling_plan")
    if not isinstance(plan, dict) or plan.get("verified") is not True:
        raise ValueError(
            "measurement has no verified Kirei Benchmark Sampling plan; the benchmark must "
            "generate SAMPLER/SIGMAS from scenarios.json instead of reusing workflow widgets"
        )
    expected = _expected_sampling_plan(scenario, recipe)
    for key, value in expected.items():
        got = plan.get(key)
        if isinstance(value, float):
            matches = _close(got, value)
        else:
            matches = got == value
        if not matches:
            raise ValueError(
                f"sampling plan mismatch for {key}: measured {got!r}, expected {value!r} "
                f"for scenario {scenario['id']!r}"
            )

    if recipe.get("expects_vdn_runtime") and plan.get("sampler_name") == "res_multistep":
        raise ValueError(
            "res_multistep is a base-model sampler and is not valid for the OpenVDN "
            "Stage-DMD/Stage-B benchmark; use Euler + simple"
        )
    return plan


def _validate_recipe(measurement: dict, scenario: dict, recipe: dict):
    expected_steps = int(recipe["steps"])
    if int(scenario["steps"]) != expected_steps:
        if scenario.get("active") or scenario.get("comparable"):
            raise ValueError(
                f"active scenario {scenario['id']!r} uses {scenario['steps']} steps, "
                f"but recipe {scenario['recipe_id']!r} requires {expected_steps}"
            )

    plan = _validate_sampling_plan(measurement, scenario, recipe)
    runtime = measurement.get("runtime_report")
    expects_vdn = bool(recipe.get("expects_vdn_runtime", False))
    if expects_vdn and not isinstance(runtime, dict):
        raise ValueError(
            f"scenario {scenario['id']!r} uses VDN recipe {scenario['recipe_id']!r} but "
            "the measurement has no VDN Runtime Report"
        )
    if not expects_vdn and isinstance(runtime, dict):
        raise ValueError(
            f"scenario {scenario['id']!r} is a non-VDN control but the measurement carries VDN state"
        )

    if isinstance(runtime, dict):
        _validate_vdn_runtime_label(runtime, scenario)
        expected_turbo_steps = recipe.get("runtime_turbo_num_steps")
        checkpoint_recipe = runtime.get("checkpoint_recipe")
        if expected_turbo_steps is not None:
            got = checkpoint_recipe.get("turbo_num_steps") if isinstance(checkpoint_recipe, dict) else None
            if got is None or int(got) != int(expected_turbo_steps):
                raise ValueError(
                    f"runtime checkpoint turbo_num_steps={got!r}, expected {expected_turbo_steps} "
                    f"for recipe {scenario['recipe_id']!r}"
                )

        adapter_state = _runtime_adapters(runtime)
        required = list(recipe.get("required_adapters", []))
        if required and expects_vdn and adapter_state is not None and adapter_state["active"] is not None:
            if set(adapter_state["active"]) != set(required):
                raise ValueError(
                    f"runtime adapters {adapter_state['active']} != required {required} "
                    f"for recipe {scenario['recipe_id']!r}"
                )
            strength = recipe.get("required_adapter_strength")
            if strength is not None:
                for name in required:
                    got = adapter_state["strengths"].get(name)
                    if not _close(got, strength):
                        raise ValueError(
                            f"adapter {name!r} strength={got!r}, expected {strength} for "
                            f"recipe {scenario['recipe_id']!r}"
                        )

    sampling = measurement.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("benchmark measurement does not expose the model's MiniMax-H3 shifts")
    got_v, got_a = sampling.get("video_shift"), sampling.get("audio_shift")
    if not _close(got_v, plan["video_shift"]) or not _close(got_a, plan["audio_shift"]):
        raise ValueError(
            f"model sampling shifts video/audio={got_v}/{got_a}, expected "
            f"{plan['video_shift']}/{plan['audio_shift']} for recipe {scenario['recipe_id']!r}"
        )
    return plan


def build_result(
    measurement: dict,
    scenario: dict,
    recipe: dict,
    *,
    seed: int,
    prompt_hash: str,
    quality_status: str,
    scheduler_label: str | None = None,
    end_to_end_seconds: float | None = None,
):
    if quality_status not in QUALITY_STATUSES:
        raise ValueError(f"quality_status must be one of {sorted(QUALITY_STATUSES)}")
    scenario_id = str(measurement.get("scenario_id", ""))
    if scenario_id != scenario["id"]:
        raise ValueError(
            f"measurement scenario_id={scenario_id!r} does not match selected scenario {scenario['id']!r}"
        )
    plan = _validate_recipe(measurement, scenario, recipe)

    row = {
        "scenario_id": scenario["id"],
        "product_group": scenario.get("product_group"),
        "technical_group": scenario.get("technical_group"),
        "quality_target": scenario["quality_target"],
        "recipe_id": scenario["recipe_id"],
        "recipe_label": recipe.get("label"),
        "quality_status": quality_status,
        "quality_gate_required": bool(scenario.get("quality_gate_required", False)),
        "comparable": bool(scenario.get("comparable", True)),
        "width": int(scenario["width"]),
        "height": int(scenario["height"]),
        "frames": int(scenario["frames"]),
        "steps": int(plan["steps"]),
        "sampler_name": plan["sampler_name"],
        "scheduler_name": plan["scheduler_name"],
        "denoise": float(plan["denoise"]),
        "seed": int(seed),
        "scheduler_label": str(scheduler_label or f"{plan['sampler_name']}/{plan['scheduler_name']}"),
        "scheduler_family": str(recipe["scheduler_family"]),
        "video_shift": float(plan["video_shift"]),
        "audio_shift": float(plan["audio_shift"]),
        "prompt_hash": str(prompt_hash),
        "run_kind": str(measurement.get("run_kind", "warm")),
        "sampler_seconds": float(measurement["sampler_seconds"]),
        "peak_vram_bytes": measurement.get("peak_vram_bytes"),
        "sampling_plan": plan,
        "sampling": measurement.get("sampling"),
        "runtime_report": measurement.get("runtime_report"),
    }
    if end_to_end_seconds is not None:
        row["end_to_end_seconds"] = float(end_to_end_seconds)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Merge a verified Kirei benchmark measurement with its scenario recipe."
    )
    parser.add_argument("measurement", type=Path, help="JSON emitted by Kirei Benchmark End")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt-hash", required=True)
    parser.add_argument(
        "--scheduler",
        dest="scheduler_label",
        help="Optional human-readable label. Actual sampler/scheduler are read from the verified measurement.",
    )
    parser.add_argument("--quality-status", choices=sorted(QUALITY_STATUSES), default="pending")
    parser.add_argument("--end-to-end-seconds", type=float)
    args = parser.parse_args()

    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    _payload, scenarios, recipes = load_scenarios(args.scenarios)
    scenario_id = str(measurement.get("scenario_id", ""))
    if scenario_id not in scenarios:
        raise SystemExit(f"unknown scenario_id {scenario_id!r}")
    scenario = scenarios[scenario_id]
    recipe_id = scenario.get("recipe_id")
    if recipe_id not in recipes:
        raise SystemExit(f"scenario {scenario_id!r} references unknown recipe {recipe_id!r}")
    row = build_result(
        measurement,
        scenario,
        recipes[recipe_id],
        seed=args.seed,
        prompt_hash=args.prompt_hash,
        quality_status=args.quality_status,
        scheduler_label=args.scheduler_label,
        end_to_end_seconds=args.end_to_end_seconds,
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
