from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = {
    "scenario_id",
    "comparison_group",
    "width",
    "height",
    "frames",
    "steps",
    "seed",
    "scheduler",
    "sampler_seconds",
}


def load_jsonl(path: Path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(REQUIRED - set(row))
        if missing:
            raise ValueError(f"{path}:{line_no}: missing fields {missing}")
        rows.append(row)
    return rows


def invariant(row):
    return (
        int(row["width"]),
        int(row["height"]),
        int(row["frames"]),
        int(row["steps"]),
        int(row["seed"]),
        str(row["scheduler"]),
    )


def summarize(rows):
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario_id"]].append(row)

    summary = {}
    for scenario, items in by_scenario.items():
        groups = {item["comparison_group"] for item in items}
        invariants = {invariant(item) for item in items}
        if len(groups) != 1:
            raise ValueError(f"scenario {scenario!r} spans comparison groups {sorted(groups)}")
        if len(invariants) != 1:
            raise ValueError(f"scenario {scenario!r} mixes benchmark objectives: {sorted(invariants)}")
        samples = [float(item["sampler_seconds"]) for item in items]
        e2e = [float(item["end_to_end_seconds"]) for item in items if item.get("end_to_end_seconds") is not None]
        vram = [int(item["peak_vram_bytes"]) for item in items if item.get("peak_vram_bytes") is not None]
        summary[scenario] = {
            "comparison_group": next(iter(groups)),
            "invariant": next(iter(invariants)),
            "runs": len(samples),
            "sampler_median": statistics.median(samples),
            "sampler_min": min(samples),
            "sampler_max": max(samples),
            "end_to_end_median": statistics.median(e2e) if e2e else None,
            "peak_vram_max": max(vram) if vram else None,
        }
    return summary


def comparisons(summary):
    grouped = defaultdict(list)
    for scenario, item in summary.items():
        grouped[item["comparison_group"]].append((scenario, item))

    out = {}
    for group, items in sorted(grouped.items()):
        invariants = {item["invariant"] for _, item in items}
        if len(invariants) != 1:
            raise ValueError(
                f"comparison_group {group!r} mixes resolution/frames/steps/seed/scheduler: {sorted(invariants)}"
            )
        ranked = sorted(items, key=lambda pair: pair[1]["sampler_median"])
        best = ranked[0][1]["sampler_median"]
        out[group] = [
            {
                "scenario_id": scenario,
                "sampler_median": item["sampler_median"],
                "relative_to_fastest": item["sampler_median"] / best,
                "runs": item["runs"],
            }
            for scenario, item in ranked
        ]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize VDN-H3 benchmark JSONL without comparing different quality objectives."
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    summary = summarize(rows)
    payload = {"summary": summary, "comparisons": comparisons(summary)}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        args.json_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
