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
    "seed",
    "scheduler",
    "prompt_hash",
    "sampler_seconds",
}

QUALITY_STATUSES = {"pending", "qualified", "failed", "diagnostic"}


def _normalize(row: dict) -> dict:
    row = dict(row)
    # v2 compatibility: old files had one comparison_group and a free-form recipe.
    row.setdefault("recipe_id", row.get("recipe", "legacy"))
    row.setdefault("product_group", None)
    row.setdefault("technical_group", row.get("comparison_group"))
    row.setdefault("scheduler_family", row.get("scheduler", "unknown"))
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


def scenario_invariant(row):
    """Fields that must never vary between repeated runs of one scenario."""
    return (
        int(row["width"]),
        int(row["height"]),
        int(row["frames"]),
        int(row["steps"]),
        int(row["seed"]),
        str(row["scheduler"]),
        str(row["scheduler_family"]),
        str(row["prompt_hash"]),
        str(row["quality_target"]),
        str(row["recipe_id"]),
        row.get("video_shift"),
        row.get("audio_shift"),
    )


def product_invariant(item):
    """Same output objective; each model is allowed to use its trained NFE recipe."""
    row = item["representative"]
    return (
        int(row["width"]),
        int(row["height"]),
        int(row["frames"]),
        int(row["seed"]),
        str(row["scheduler_family"]),
        str(row["prompt_hash"]),
        str(row["quality_target"]),
        row.get("video_shift"),
        row.get("audio_shift"),
    )


def technical_invariant(item):
    """Strict same-work comparison used for BF16/INT8/max-speed engineering claims."""
    row = item["representative"]
    return (
        int(row["width"]),
        int(row["height"]),
        int(row["frames"]),
        int(row["steps"]),
        int(row["seed"]),
        str(row["scheduler"]),
        str(row["scheduler_family"]),
        str(row["prompt_hash"]),
        str(row["quality_target"]),
        str(row["recipe_id"]),
        row.get("video_shift"),
        row.get("audio_shift"),
    )


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


def _comparison(summary, group_field: str, invariant_fn, mode: str):
    grouped = defaultdict(list)
    for scenario, item in summary.items():
        group = item.get(group_field)
        if not item["comparable"] or not group:
            continue
        grouped[group].append((scenario, item))

    out = {}
    for group, items in sorted(grouped.items()):
        invariants = {invariant_fn(item) for _, item in items}
        if len(invariants) != 1:
            raise ValueError(
                f"{mode} group {group!r} mixes incompatible benchmark objectives: {sorted(invariants)}"
            )
        ranked = sorted(items, key=lambda pair: pair[1]["sampler_median"])
        best = ranked[0][1]["sampler_median"]
        claim_eligible = all(
            (not item["quality_gate_required"]) or item["quality_status"] == "qualified"
            for _, item in ranked
        )
        out[group] = {
            "mode": mode,
            "speed_claim_eligible": claim_eligible,
            "quality_statuses": {scenario: item["quality_status"] for scenario, item in ranked},
            "ranking": [
                {
                    "scenario_id": scenario,
                    "recipe_id": item["recipe_id"],
                    "steps": item["steps"],
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
    return _comparison(summary, "product_group", product_invariant, "recipe_faithful_product")


def technical_comparisons(summary):
    return _comparison(summary, "technical_group", technical_invariant, "same_nfe_technical")


def comparisons(summary):
    """Compatibility wrapper for callers that only need both comparison classes."""
    return {
        "product": product_comparisons(summary),
        "technical": technical_comparisons(summary),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize VDN-H3 benchmark JSONL. Product rankings compare each model at its "
            "intended recipe (Turbo 4 NFE vs VDN-DMD 8 NFE). Technical rankings compare "
            "VDN execution variants at identical NFE. Quality gates control speed claims."
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
