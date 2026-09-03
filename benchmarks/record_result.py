from __future__ import annotations

import argparse
import json
from pathlib import Path

QUALITY_STATUSES = {"pending", "qualified", "failed", "diagnostic"}


def load_scenarios(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = {item["id"]: item for item in payload["scenarios"]}
    return payload, scenarios


def build_result(
    measurement: dict,
    scenario: dict,
    rules: dict,
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

    declared = int(rules.get("checkpoint_declared_turbo_steps", 0) or 0)
    if (
        scenario.get("active")
        and scenario.get("quality_target", "").startswith("checkpoint_declared_distilled8")
        and declared
        and int(scenario["steps"]) != declared
    ):
        raise ValueError(
            f"active distilled scenario {scenario['id']!r} uses {scenario['steps']} steps, "
            f"but benchmark rules declare {declared}"
        )

    runtime = measurement.get("runtime_report")
    runtime_declared = None
    if isinstance(runtime, dict):
        recipe = runtime.get("checkpoint_recipe")
        if isinstance(recipe, dict):
            value = recipe.get("turbo_num_steps")
            if value is not None:
                runtime_declared = int(value)
                if int(scenario["steps"]) != runtime_declared:
                    raise ValueError(
                        f"runtime checkpoint declares turbo_num_steps={runtime_declared}, but scenario "
                        f"{scenario['id']!r} requests steps={scenario['steps']}"
                    )

    row = {
        "scenario_id": scenario["id"],
        "comparison_group": scenario["comparison_group"],
        "quality_target": scenario["quality_target"],
        "recipe": scenario["recipe"],
        "quality_status": quality_status,
        "quality_gate_required": bool(scenario.get("quality_gate_required", False)),
        "comparable": bool(scenario.get("comparable", True)),
        "width": int(scenario["width"]),
        "height": int(scenario["height"]),
        "frames": int(scenario["frames"]),
        "steps": int(scenario["steps"]),
        "seed": int(seed),
        "scheduler": str(scheduler),
        "prompt_hash": str(prompt_hash),
        "checkpoint_turbo_num_steps": runtime_declared if runtime_declared is not None else declared or None,
        "run_kind": str(measurement.get("run_kind", "warm")),
        "sampler_seconds": float(measurement["sampler_seconds"]),
        "peak_vram_bytes": measurement.get("peak_vram_bytes"),
        "runtime_report": runtime,
    }
    if end_to_end_seconds is not None:
        row["end_to_end_seconds"] = float(end_to_end_seconds)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Merge a Kirei Benchmark End measurement with a validated benchmark scenario."
    )
    parser.add_argument("measurement", type=Path, help="JSON emitted by Kirei Benchmark End")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--prompt-hash", required=True)
    parser.add_argument(
        "--quality-status",
        choices=sorted(QUALITY_STATUSES),
        default="pending",
    )
    parser.add_argument("--end-to-end-seconds", type=float)
    args = parser.parse_args()

    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    payload, scenarios = load_scenarios(args.scenarios)
    scenario_id = str(measurement.get("scenario_id", ""))
    if scenario_id not in scenarios:
        raise SystemExit(f"unknown scenario_id {scenario_id!r}")
    row = build_result(
        measurement,
        scenarios[scenario_id],
        payload["rules"],
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
