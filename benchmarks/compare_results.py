from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CORE_REQUIRED = {
    "scenario_id",
    "quality_target",
    "quality_status",
    "quality_gate_required",
    "comparable",
    "width",
    "height",
    "frames",
    "steps",
    "sampler_name",
    "scheduler_name",
    "denoise",
    "seed",
    "prompt_hash",
    "sampler_seconds",
}

QUALITY_STATUSES = {"pending", "qualified", "failed", "diagnostic"}


def _normalize(row: dict) -> dict:
    row = dict(row)
    row.setdefault("recipe_id", row.get("recipe", "legacy"))
    row.setdefault("product_group", None)
    row.setdefault("technical_group", row.get("comparison_group"))
    row.setdefault("sampler_name", row.get("sampler", "legacy_unknown"))
    row.setdefault("scheduler_name", row.get("scheduler", "legacy_unknown"))
    row.setdefault("denoise", 1.0)
    row.setdefault("scheduler_family", row.get("scheduler_name", "unknown"))
    row.setdefault("video_shift", None)
    row.setdefault("audio_shift", None)
    return row


def load_jsonl(path: Path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _normalize(json.loads(line))
        missing = sorted(CORE_REQUIRED - set(row))
        if missing:
            raise ValueError(f"{path}:{line_no}: missing fields {missing}")
        if row["quality_status"] not in QUALITY_STATUSES:
            raise ValueError(
                f"{path}:{line_no}: quality_status must be one of {sorted(QUALITY_STATUSES)}, "
                f"got {row['quality_status']!r}"
            )
        rows.append(row)
    return rows


def _trajectory(row):
    return (
        int(row["steps"]),
        str(row["sampler_name"]),
        str(row["scheduler_name"]),
        float(row["denoise"]),
        str(row["scheduler_family"]),
        row.get("video_shift"),
        row.get("audio_shift"),
    )


def scenario_invariant(row):
    return (
        int(row["width"]),
        int(row["height"]),
        int(row["frames"]),
        *_trajectory(row),
        int(row["seed"]),
        str(row["prompt_hash"]),
        str(row["quality_target"]),
        str(row["recipe_id"]),
    )


def product_invariant(item):
    row = item["representative"]
    return (
        int(row["width"]),
        int(row["height"]),
        int(row["frames"]),
        *_trajectory(row),
        int(row["seed"]),
        str(row["prompt_hash"]),
        str(row["quality_target"]),
    )


def technical_invariant(item):
    row = item["representative"]
    return (*product_invariant(item), str(row["recipe_id"]))


def summarize(rows):
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario_id"]].append(row)

    summary = {}
    for scenario, items in by_scenario.items():
        invariants = {scenario_invariant(item) for item in items}
        if len(invariants) != 1:
            raise ValueError(f"scenario {scenario!r} mixes benchmark objectives: {sorted(invariants)}")
        for field in ("comparable", "quality_gate_required", "quality_status", "product_group", "technical_group"):
            values = {str(item.get(field)) for item in items}
            if len(values) != 1:
                raise ValueError(f"scenario {scenario!r} mixes {field}: {sorted(values)}")

        samples = [float(item["sampler_seconds"]) for item in items]
        e2e = [float(item["end_to_end_seconds"]) for item in items if item.get("end_to_end_seconds") is not None]
        vram = [int(item["peak_vram_bytes"]) for item in items if item.get("peak_vram_bytes") is not None]
        representative = items[0]
        summary[scenario] = {
            "representative": representative,
            "product_group": representative.get("product_group"),
            "technical_group": representative.get("technical_group"),
            "recipe_id": representative["recipe_id"],
            "steps": int(representative["steps"]),
            "sampler_name": representative["sampler_name"],
            "scheduler_name": representative["scheduler_name"],
            "denoise": float(representative["denoise"]),
            "comparable": bool(representative["comparable"]),
            "quality_gate_required": bool(representative["quality_gate_required"]),
            "quality_status": str(representative["quality_status"]),
            "runs": len(samples),
            "sampler_median": statistics.median(samples),
            "sampler_min": min(samples),
            "sampler_max": max(samples),
            "end_to_end_median": statistics.median(e2e) if e2e else None,
            "peak_vram_max": max(vram) if vram else None,
        }
    return summary


def _comparison(summary, group_field: str, invariant_fn, mode: str, *, min_items: int = 1):
    grouped = defaultdict(list)
    for scenario, item in summary.items():
        group = item.get(group_field)
        if not item["comparable"] or not group:
            continue
        grouped[group].append((scenario, item))

    out = {}
    for group, items in sorted(grouped.items()):
        if len(items) < min_items:
            continue
        invariants = {invariant_fn(item) for _, item in items}
        if len(invariants) != 1:
            raise ValueError(
                f"{mode} group {group!r} mixes incompatible geometry/sampler/scheduler/NFE/"
                f"denoise/seed/prompt/quality objectives: {sorted(invariants)}"
            )
        ranked = sorted(items, key=lambda pair: pair[1]["sampler_median"])
        best = ranked[0][1]["sampler_median"]
        claim_eligible = all(
            (not item["quality_gate_required"]) or item["quality_status"] == "qualified"
            for _, item in ranked
        )
        out[group] = {
            "mode": mode,
            "trajectory": {
                "steps": ranked[0][1]["steps"],
                "sampler_name": ranked[0][1]["sampler_name"],
                "scheduler_name": ranked[0][1]["scheduler_name"],
                "denoise": ranked[0][1]["denoise"],
            },
            "speed_claim_eligible": claim_eligible,
            "quality_statuses": {scenario: item["quality_status"] for scenario, item in ranked},
            "ranking": [
                {
                    "scenario_id": scenario,
                    "recipe_id": item["recipe_id"],
                    "steps": item["steps"],
                    "sampler_name": item["sampler_name"],
                    "scheduler_name": item["scheduler_name"],
                    "denoise": item["denoise"],
                    "sampler_median": item["sampler_median"],
                    "relative_to_fastest": item["sampler_median"] / best,
                    "runs": item["runs"],
                    "quality_status": item["quality_status"],
                    "same_quality_speed_claim": bool(
                        claim_eligible
                        and ((not item["quality_gate_required"]) or item["quality_status"] == "qualified")
                    ),
                }
                for scenario, item in ranked
            ],
        }
    return out


def product_comparisons(summary):
    return _comparison(summary, "product_group", product_invariant, "same_objective_product", min_items=2)


def technical_comparisons(summary):
    # Product groups intentionally contain Larry and VDN controls.  Technical speed
    # comparisons must never mix those recipes, so partition a nominal technical group
    # by recipe_id and omit singletons (Larry becomes a product control, while the VDN
    # BF16/INT8/FP8/max_speed variants remain a meaningful same-recipe group).
    partitioned = {}
    for scenario, item in summary.items():
        group = item.get("technical_group")
        if not group:
            partitioned[scenario] = item
            continue
        clone = dict(item)
        clone["technical_partition"] = f"{group}::{item['recipe_id']}"
        partitioned[scenario] = clone
    return _comparison(
        partitioned,
        "technical_partition",
        technical_invariant,
        "same_recipe_technical",
        min_items=2,
    )


def comparisons(summary):
    return {"product": product_comparisons(summary), "technical": technical_comparisons(summary)}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize verified VDN-H3 benchmark JSONL. Comparable rows must share the exact "
            "sampler trajectory (Euler/simple/NFE/denoise); technical rankings are isolated by recipe."
        )
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    summary = summarize(rows)
    payload = {
        "summary": summary,
        "product_comparisons": product_comparisons(summary),
        "technical_comparisons": technical_comparisons(summary),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    print(text)
    if args.json_path:
        args.json_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
