"""ComfyUI nodes for recipe-aware sampler benchmarks.

Use ``Kirei Benchmark Scenario`` as the single source of truth for the test geometry and
recipe, then place ``Kirei Benchmark Start`` immediately before the sampler's MODEL input
and connect the sampler LATENT directly to ``Kirei Benchmark End.after``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


_TOKEN_TYPE = "KIREI_BENCHMARK_TOKEN"
_SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "scenarios.json"


def _scenario_payload():
    try:
        payload = json.loads(_SCENARIOS_PATH.read_text(encoding="utf-8"))
        scenarios = {item["id"]: item for item in payload.get("scenarios", [])}
        return payload, scenarios
    except Exception:
        return {"schema_version": 0, "recipes": {}}, {}


def _active_scenario_ids():
    _payload, scenarios = _scenario_payload()
    active = [key for key, item in scenarios.items() if item.get("active")]
    return active or ["benchmark"]


def _resolve_scenario(scenario_id: str):
    payload, scenarios = _scenario_payload()
    scenario = scenarios.get(str(scenario_id))
    if scenario is None:
        raise ValueError(f"unknown benchmark scenario {scenario_id!r}")
    if not scenario.get("active"):
        raise ValueError(f"benchmark scenario {scenario_id!r} is not active")
    recipe_id = scenario.get("recipe_id")
    recipe = (payload.get("recipes") or {}).get(recipe_id)
    if not isinstance(recipe, dict):
        raise ValueError(f"benchmark scenario {scenario_id!r} references unknown recipe {recipe_id!r}")
    return payload, scenario, recipe


def _scenario_runtime_controls(spec: dict, recipe: dict):
    expects_vdn = bool(recipe.get("expects_vdn_runtime", False))
    if not expects_vdn:
        return "n/a", "n/a", False
    profile = str(spec.get("profile", "auto"))
    projection = str(spec.get("projection_precision", "auto"))
    apply_turbo = bool(
        spec.get("apply_turbo_adapter", spec.get("recipe_id") == "vdn_stage_dmd_8")
    )
    return profile, projection, apply_turbo


def _model_device(model: Any) -> torch.device:
    try:
        return torch.device(getattr(model, "load_device", "cpu"))
    except Exception:
        return torch.device("cpu")


def _model_sampling(model: Any):
    candidates = [
        getattr(model, "model_sampling", None),
        getattr(getattr(model, "model", None), "model_sampling", None),
    ]
    inner = getattr(model, "inner_model", None)
    if inner is not None:
        candidates.extend(
            [
                getattr(inner, "model_sampling", None),
                getattr(getattr(inner, "inner_model", None), "model_sampling", None),
            ]
        )
    return next((item for item in candidates if item is not None), None)


def _sampling_snapshot(model: Any) -> dict:
    sampling = _model_sampling(model)
    options = getattr(model, "model_options", {}) or {}
    transformer_options = options.get("transformer_options", {}) if isinstance(options, dict) else {}

    def value(name, option_name=None):
        raw = getattr(sampling, name, None) if sampling is not None else None
        if raw is None and option_name:
            raw = transformer_options.get(option_name)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "class": type(sampling).__name__ if sampling is not None else None,
        "video_shift": value("shift", "minimax_h3_sigma_shift_video"),
        "audio_shift": value("audio_shift", "minimax_h3_sigma_shift_audio"),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _cuda_start(device: torch.device, reset_peak: bool) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_start_bytes": None, "reserved_start_bytes": None}
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_start_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_start_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _cuda_end(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_end_bytes": None,
            "reserved_end_bytes": None,
            "peak_vram_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_end_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_end_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


class KireiBenchmarkScenario:
    """Expose one active benchmark scenario as connectable workflow values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenario": (
                    _active_scenario_ids(),
                    {
                        "tooltip": (
                            "Select once, then wire geometry/steps/shifts into the workflow. "
                            "The VDN profile/projection/turbo outputs show the exact Apply VDN settings."
                        )
                    },
                )
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT",
        "STRING", "STRING", "BOOLEAN", "STRING",
    )
    RETURN_NAMES = (
        "scenario_id",
        "recipe_id",
        "steps",
        "width",
        "height",
        "frames",
        "video_shift",
        "audio_shift",
        "vdn_profile",
        "projection_precision",
        "apply_turbo_adapter",
        "scenario_json",
    )
    FUNCTION = "resolve"
    CATEGORY = "model_patches/video/benchmark"
    DESCRIPTION = "Resolve an active benchmark scenario into connectable recipe, geometry and VDN-control values."

    def resolve(self, scenario):
        payload, spec, recipe = _resolve_scenario(str(scenario))
        profile, projection, apply_turbo = _scenario_runtime_controls(spec, recipe)
        merged = {
            "schema_version": int(payload.get("schema_version", 0)),
            "scenario": spec,
            "recipe": recipe,
            "runtime_controls": {
                "vdn_profile": profile,
                "projection_precision": projection,
                "apply_turbo_adapter": apply_turbo,
            },
        }
        return (
            str(spec["id"]),
            str(spec["recipe_id"]),
            int(spec["steps"]),
            int(spec["width"]),
            int(spec["height"]),
            int(spec["frames"]),
            float(recipe["video_shift"]),
            float(recipe["audio_shift"]),
            profile,
            projection,
            apply_turbo,
            json.dumps(merged, indent=2, sort_keys=True, default=str),
        )


class KireiBenchmarkStart:
    @classmethod
    def INPUT_TYPES(cls):
        default = _active_scenario_ids()[0]
        return {
            "required": {
                "model": ("MODEL",),
                "scenario_id": (
                    "STRING",
                    {
                        "default": default,
                        "tooltip": (
                            "Connect Kirei Benchmark Scenario.scenario_id here. The id is validated "
                            "against benchmarks/scenarios.json before timing starts."
                        ),
                    },
                ),
            },
            "optional": {"reset_peak_vram": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("MODEL", _TOKEN_TYPE)
    RETURN_NAMES = ("model", "benchmark_token")
    FUNCTION = "start"
    CATEGORY = "model_patches/video/benchmark"
    DESCRIPTION = "Start a recipe-aware sampler benchmark and record the model's actual H3 sampling shifts."

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def start(self, model, scenario_id, reset_peak_vram=True):
        payload, scenario, recipe = _resolve_scenario(str(scenario_id))
        device = _model_device(model)
        _sync(device)
        memory = _cuda_start(device, bool(reset_peak_vram))
        token = {
            "scenario_id": str(scenario_id),
            "scenario_schema_version": int(payload.get("schema_version", 0)),
            "scenario_spec": scenario,
            "recipe_spec": recipe,
            "sampling": _sampling_snapshot(model),
            "device": str(device),
            "started_ns": time.perf_counter_ns(),
            "reset_peak_vram": bool(reset_peak_vram),
            **memory,
        }
        return model, token


class KireiBenchmarkEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "benchmark_token": (_TOKEN_TYPE,),
                "after": (
                    "*",
                    {
                        "tooltip": (
                            "Connect the sampler LATENT directly here. Connecting VAE/video output "
                            "would intentionally include decode/encoding in the measured interval."
                        )
                    },
                ),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect the same sampled MODEL. VDN scenarios then embed the Runtime "
                            "Report so checkpoint recipe/profile/precision can be validated."
                        )
                    },
                ),
                "run_kind": (["cold", "warm"], {"default": "warm"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("measurement_json",)
    FUNCTION = "end"
    CATEGORY = "model_patches/video/benchmark"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Close the sampler benchmark after GPU synchronization and emit wall time, peak VRAM, "
        "scenario/recipe, actual sampling shifts and optional VDN runtime metadata."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def end(self, benchmark_token, after, model=None, run_kind="warm"):
        del after
        if not isinstance(benchmark_token, dict) or "started_ns" not in benchmark_token:
            raise TypeError("Kirei Benchmark End received an invalid benchmark token")
        device = torch.device(benchmark_token.get("device", "cpu"))
        _sync(device)
        finished_ns = time.perf_counter_ns()
        elapsed = (finished_ns - int(benchmark_token["started_ns"])) / 1_000_000_000.0
        payload = {
            "scenario_id": str(benchmark_token.get("scenario_id", "benchmark")),
            "scenario_schema_version": benchmark_token.get("scenario_schema_version"),
            "scenario_spec": benchmark_token.get("scenario_spec"),
            "recipe_spec": benchmark_token.get("recipe_spec"),
            "sampling": benchmark_token.get("sampling"),
            "run_kind": str(run_kind),
            "device": str(device),
            "sampler_seconds": elapsed,
            "started_ns": int(benchmark_token["started_ns"]),
            "finished_ns": int(finished_ns),
            "reset_peak_vram": bool(benchmark_token.get("reset_peak_vram", False)),
            "allocated_start_bytes": benchmark_token.get("allocated_start_bytes"),
            "reserved_start_bytes": benchmark_token.get("reserved_start_bytes"),
            **_cuda_end(device),
        }
        if model is not None:
            try:
                from .benchmark import runtime_snapshot
                payload["runtime_report"] = runtime_snapshot(model)
            except RuntimeError:
                payload["runtime_report"] = None
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "KireiBenchmarkScenario": KireiBenchmarkScenario,
    "KireiBenchmarkStart": KireiBenchmarkStart,
    "KireiBenchmarkEnd": KireiBenchmarkEnd,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiBenchmarkScenario": "Kirei Benchmark Scenario",
    "KireiBenchmarkStart": "Kirei Benchmark Start",
    "KireiBenchmarkEnd": "Kirei Benchmark End",
}


__all__ = [
    "KireiBenchmarkScenario",
    "KireiBenchmarkStart",
    "KireiBenchmarkEnd",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
