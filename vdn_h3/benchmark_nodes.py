"""ComfyUI nodes for recipe-aware sampler benchmarks.

The benchmark path deliberately owns the sampler and sigma schedule.  This prevents a
saved workflow widget from silently leaking a different sampler (for example
``res_multistep``) into an OpenVDN Stage-DMD run that is supposed to use Euler + simple.

Recommended wiring::

    Kirei Benchmark Scenario
          | scenario_id
          v
    Kirei Benchmark Sampling <--- MODEL
          | SAMPLER / SIGMAS --------> SamplerCustomAdvanced
          | recipe_token
          v
    Kirei Benchmark Start <----------- same MODEL
          | MODEL --------------------> SamplerCustomAdvanced
          | benchmark_token
          v
    Kirei Benchmark End <------------- sampler LATENT

``Kirei Benchmark Start`` refuses an unverified recipe token, so benchmark measurements
cannot be recorded from a hand-configured sampler by accident.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .nodes import _PROFILES
from .projection import PROJECTION_PRECISIONS


_TOKEN_TYPE = "KIREI_BENCHMARK_TOKEN"
_RECIPE_TOKEN_TYPE = "KIREI_BENCHMARK_RECIPE"
_PROFILE_TYPE = list(_PROFILES)
_PROJECTION_TYPE = ["auto", *PROJECTION_PRECISIONS]
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


def _sampling_plan(payload: dict, spec: dict, recipe: dict) -> dict:
    steps = int(spec["steps"])
    recipe_steps = int(recipe["steps"])
    if steps != recipe_steps:
        raise ValueError(
            f"active benchmark scenario {spec['id']!r} requests {steps} steps but recipe "
            f"{spec['recipe_id']!r} requires {recipe_steps}"
        )
    return {
        "verified": True,
        "scenario_schema_version": int(payload.get("schema_version", 0)),
        "scenario_id": str(spec["id"]),
        "recipe_id": str(spec["recipe_id"]),
        "sampler_name": str(recipe["sampler_name"]),
        "scheduler_name": str(recipe["scheduler_name"]),
        "steps": steps,
        "denoise": float(recipe.get("denoise", 1.0)),
        "video_shift": float(recipe["video_shift"]),
        "audio_shift": float(recipe["audio_shift"]),
    }


def _model_device(model: Any) -> torch.device:
    try:
        return torch.device(getattr(model, "load_device", "cpu"))
    except Exception:
        return torch.device("cpu")


def _model_sampling(model: Any):
    getter = getattr(model, "get_model_object", None)
    if callable(getter):
        try:
            value = getter("model_sampling")
            if value is not None:
                return value
        except Exception:
            pass
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


def _close(a, b, tol=1e-6):
    return a is not None and b is not None and math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0.0)


def _validate_model_shifts(snapshot: dict, plan: dict) -> None:
    # MiniMax-H3 should expose both values on current ComfyUI. Refuse to benchmark an
    # unknown/different clock because it changes the actual denoising trajectory.
    if not _close(snapshot.get("video_shift"), plan["video_shift"]):
        raise ValueError(
            f"benchmark model video shift={snapshot.get('video_shift')!r}, expected "
            f"{plan['video_shift']} for scenario {plan['scenario_id']!r}"
        )
    if not _close(snapshot.get("audio_shift"), plan["audio_shift"]):
        raise ValueError(
            f"benchmark model audio shift={snapshot.get('audio_shift')!r}, expected "
            f"{plan['audio_shift']} for scenario {plan['scenario_id']!r}"
        )


def _build_sampler_and_sigmas(model, plan: dict):
    """Use the same current Comfy primitives as KSamplerSelect + BasicScheduler."""
    import comfy.samplers

    sampler_name = plan["sampler_name"]
    scheduler_name = plan["scheduler_name"]
    if sampler_name not in comfy.samplers.SAMPLER_NAMES:
        raise ValueError(f"ComfyUI does not provide benchmark sampler {sampler_name!r}")
    if scheduler_name not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError(f"ComfyUI does not provide benchmark scheduler {scheduler_name!r}")

    sampler = comfy.samplers.sampler_object(sampler_name)
    sampling = _model_sampling(model)
    if sampling is None:
        raise RuntimeError("benchmark MODEL does not expose model_sampling")

    steps = int(plan["steps"])
    denoise = float(plan["denoise"])
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
            return sampler, sigmas
        total_steps = int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler_name, total_steps).cpu()
    sigmas = sigmas[-(steps + 1):]
    return sampler, sigmas


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
                            "Select once, then wire geometry and VDN controls into the workflow. "
                            "Use Kirei Benchmark Sampling for the actual SAMPLER/SIGMAS."
                        )
                    },
                )
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT",
        "STRING", "STRING", "FLOAT", _PROFILE_TYPE, _PROJECTION_TYPE, "BOOLEAN", "FLOAT", "STRING",
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
        "sampler_name",
        "scheduler_name",
        "denoise",
        "vdn_profile",
        "projection_precision",
        "apply_turbo_adapter",
        "lora_strength",
        "scenario_json",
    )
    FUNCTION = "resolve"
    CATEGORY = "model_patches/video/benchmark"
    DESCRIPTION = "Resolve an active benchmark scenario into recipe, geometry and runtime-control values."

    def resolve(self, scenario):
        payload, spec, recipe = _resolve_scenario(str(scenario))
        plan = _sampling_plan(payload, spec, recipe)
        profile, projection, apply_turbo = _scenario_runtime_controls(spec, recipe)
        lora_strength = float(recipe.get("lora_strength", recipe.get("required_adapter_strength", 1.0)))
        merged = {
            "schema_version": int(payload.get("schema_version", 0)),
            "scenario": spec,
            "recipe": recipe,
            "sampling_plan": plan,
            "runtime_controls": {
                "vdn_profile": profile,
                "projection_precision": projection,
                "apply_turbo_adapter": apply_turbo,
                "lora_strength": lora_strength,
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
            plan["sampler_name"],
            plan["scheduler_name"],
            float(plan["denoise"]),
            profile,
            projection,
            apply_turbo,
            lora_strength,
            json.dumps(merged, indent=2, sort_keys=True, default=str),
        )


class KireiBenchmarkSampling:
    """Construct the benchmark SAMPLER and SIGMAS from the selected scenario."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scenario_id": (
                    "STRING",
                    {"default": _active_scenario_ids()[0], "tooltip": "Connect Benchmark Scenario.scenario_id."},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS", _RECIPE_TOKEN_TYPE, "STRING")
    RETURN_NAMES = ("sampler", "sigmas", "recipe_token", "sampling_plan_json")
    FUNCTION = "build"
    CATEGORY = "model_patches/video/benchmark"
    DESCRIPTION = (
        "Build the exact sampler/sigma schedule declared by the benchmark scenario. "
        "OpenVDN Stage-DMD therefore always receives Euler + simple + 8 NFE + denoise 1.0."
    )

    def build(self, model, scenario_id):
        payload, spec, recipe = _resolve_scenario(str(scenario_id))
        plan = _sampling_plan(payload, spec, recipe)
        snapshot = _sampling_snapshot(model)
        _validate_model_shifts(snapshot, plan)
        sampler, sigmas = _build_sampler_and_sigmas(model, plan)
        token = dict(plan)
        token["model_sampling"] = snapshot
        return sampler, sigmas, token, json.dumps(token, indent=2, sort_keys=True, default=str)


class KireiBenchmarkStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "recipe_token": (
                    _RECIPE_TOKEN_TYPE,
                    {
                        "tooltip": (
                            "Required token from Kirei Benchmark Sampling. This proves the sampler and "
                            "sigma schedule were generated from scenarios.json rather than inherited widgets."
                        )
                    },
                ),
            },
            "optional": {"reset_peak_vram": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("MODEL", _TOKEN_TYPE)
    RETURN_NAMES = ("model", "benchmark_token")
    FUNCTION = "start"
    CATEGORY = "model_patches/video/benchmark"
    DESCRIPTION = "Start a sampler benchmark only after a verified scenario sampling recipe has been built."

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def start(self, model, recipe_token, reset_peak_vram=True):
        if not isinstance(recipe_token, dict) or recipe_token.get("verified") is not True:
            raise TypeError("Kirei Benchmark Start requires a verified Kirei Benchmark Sampling token")
        payload, scenario, recipe = _resolve_scenario(str(recipe_token.get("scenario_id", "")))
        expected = _sampling_plan(payload, scenario, recipe)
        for key, value in expected.items():
            if recipe_token.get(key) != value:
                raise ValueError(
                    f"benchmark recipe token mismatch for {key}: {recipe_token.get(key)!r} != {value!r}"
                )

        snapshot = _sampling_snapshot(model)
        _validate_model_shifts(snapshot, expected)
        device = _model_device(model)
        _sync(device)
        memory = _cuda_start(device, bool(reset_peak_vram))
        token = {
            "scenario_id": expected["scenario_id"],
            "scenario_schema_version": expected["scenario_schema_version"],
            "scenario_spec": scenario,
            "recipe_spec": recipe,
            "sampling_plan": expected,
            "sampling": snapshot,
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
                            "Connect the same sampled MODEL. VDN scenarios embed the Runtime Report "
                            "so checkpoint recipe/profile/precision can be validated."
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
        "Close the verified sampler benchmark after GPU synchronization and emit wall time, "
        "peak VRAM, exact sampling plan and optional VDN runtime metadata."
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
            "sampling_plan": benchmark_token.get("sampling_plan"),
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
    "KireiBenchmarkSampling": KireiBenchmarkSampling,
    "KireiBenchmarkStart": KireiBenchmarkStart,
    "KireiBenchmarkEnd": KireiBenchmarkEnd,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiBenchmarkScenario": "Kirei Benchmark Scenario",
    "KireiBenchmarkSampling": "Kirei Benchmark Sampling",
    "KireiBenchmarkStart": "Kirei Benchmark Start",
    "KireiBenchmarkEnd": "Kirei Benchmark End",
}


__all__ = [
    "KireiBenchmarkScenario",
    "KireiBenchmarkSampling",
    "KireiBenchmarkStart",
    "KireiBenchmarkEnd",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
