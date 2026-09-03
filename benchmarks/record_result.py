from __future__ import annotations

import argparse
import json
from pathlib import Path

QUALITY_STATUSES = {"pending", "qualified", "failed", "diagnostic"}


def load_scenarios(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    recipes = dict(payload.get("recipes", {}))
    return payload, scenarios, recipes


def _close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


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
    """Expected resolved runtime implied by one scenario id/model variant.

    These expectations are deliberately narrower than the normal node's `auto` policy:
    a benchmark named BF16/INT8/reference/max_speed must actually execute that path or
    the measurement is mislabeled and cannot enter the result set.
    """
    variant = str(scenario.get("model_variant", ""))
    expected = {}
    if variant == "vdn_max_speed":
        expected["profile"] = "max_speed"
    elif variant == "vdn_stage_b_reference":
        expected["profile"] = "reference"
    elif variant.startswith("vdn_"):
        expected["profile"] = "auto"
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


def _validate_recipe(measurement: dict, scenario: dict, recipe: dict):
    expected_steps = int(recipe["steps"])
    if int(scenario["steps"]) != expected_steps:
        if scenario.get("active") or scenario.get("comparable"):
            raise ValueError(
                f"active scenario {scenario['id']!r} uses {scenario['steps']} steps, "
                f"but recipe {scenario['recipe_id']!r} requires {expected_steps}"
            )

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

        # Newer Runtime Reports may expose exact named-adapter state. Validate it when
        # available, while remaining compatible with older installed reports.
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
        raise ValueError(
            "benchmark measurement does not expose model sampling shifts; update/reload the "
            "Kirei Benchmark Start node before recording this result"
        )
    expected_v = float(recipe["video_shift"])
    expected_a = float(recipe["audio_shift"])
    got_v, got_a = sampling.get("video_shift"), sampling.get("audio_shift")
    if not _close(got_v, expected_v) or not _close(got_a, expected_a):
        raise ValueError(
            f"sampling shifts video/audio={got_v}/{got_a}, expected {expected_v}/{expected_a} "
            f"for recipe {scenario['recipe_id']!r}"
        )


def build_result(
    measurement: dict,
    scenario: dict,
    recipe: dict,
    *,
    seed: int,
    scheduler: str,
    prompt_hash: str,
    quality_status: str,
    end_to_end_seconds: float | None = None,
):
    if quality_status not in QUALITY_STATUSES:
        raise ValueError(f"quality_status must be one of {sorted(QUALITY_STATUSES)}")
    scenario_id = str(measurement.get("scenario_id", ""))
    if scenario_id != scenario["id"]:
        raise ValueError(
            f"measurement scenario_id={scenario_id!r} does not match selected scenario {scenario['id']!r}"
        )
    _validate_recipe(measurement, scenario, recipe)

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
        "steps": int(scenario["steps"]),
        "seed": int(seed),
        "scheduler": str(scheduler),
        "scheduler_family": str(recipe["scheduler_family"]),
        "video_shift": float(recipe["video_shift"]),
        "audio_shift": float(recipe["audio_shift"]),
        "prompt_hash": str(prompt_hash),
        "run_kind": str(measurement.get("run_kind", "warm")),
        "sampler_seconds": float(measurement["sampler_seconds"]),
        "peak_vram_bytes": measurement.get("peak_vram_bytes"),
        "sampling": measurement.get("sampling"),
        "runtime_report": measurement.get("runtime_report"),
    }
    if end_to_end_seconds is not None:
        row["end_to_end_seconds"] = float(end_to_end_seconds)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Merge a Kirei Benchmark End measurement with a validated benchmark recipe."
    )
    parser.add_argument("measurement", type=Path, help="JSON emitted by Kirei Benchmark End")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--prompt-hash", required=True)
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
        scheduler=args.scheduler,
        prompt_hash=args.prompt_hash,
        quality_status=args.quality_status,
        end_to_end_seconds=args.end_to_end_seconds,
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
