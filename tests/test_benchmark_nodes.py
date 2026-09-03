import json
import math
from types import SimpleNamespace

from vdn_h3.benchmark_nodes import (
    KireiBenchmarkEnd,
    KireiBenchmarkSampling,
    KireiBenchmarkScenario,
    KireiBenchmarkStart,
    _resolve_scenario,
    _sampling_plan,
)


def _cpu_model():
    return SimpleNamespace(
        load_device="cpu",
        model=SimpleNamespace(model_sampling=SimpleNamespace(shift=12.0, audio_shift=3.0)),
    )


def test_benchmark_scenario_exposes_canonical_vdn_sampling_controls():
    node = KireiBenchmarkScenario()
    values = node.resolve("vdn_dmd_bf16_8step_608x352_241")
    (
        scenario_id,
        recipe_id,
        steps,
        width,
        height,
        frames,
        shift_v,
        shift_a,
        sampler_name,
        scheduler_name,
        denoise,
        profile,
        projection,
        apply_turbo,
        lora_strength,
        text,
    ) = values
    assert scenario_id == "vdn_dmd_bf16_8step_608x352_241"
    assert recipe_id == "vdn_stage_dmd_8"
    assert (steps, width, height, frames) == (8, 608, 352, 241)
    assert (shift_v, shift_a) == (12.0, 3.0)
    assert (sampler_name, scheduler_name, denoise) == ("euler", "simple", 1.0)
    assert profile == "auto"
    assert projection == "bf16"
    assert apply_turbo is True
    assert lora_strength == 1.0
    payload = json.loads(text)
    assert payload["sampling_plan"]["sampler_name"] == "euler"
    assert payload["sampling_plan"]["scheduler_name"] == "simple"
    assert payload["recipe"]["required_adapters"] == ["default", "turbo"]


def test_benchmark_scenario_keeps_res_multistep_on_native_only():
    node = KireiBenchmarkScenario()
    native = node.resolve("native20_608x352_121")
    assert native[2] == 20
    assert native[8] == "res_multistep"
    assert native[9] == "simple"

    stage_b = node.resolve("vdn_stage_b_bf16_50step_1344x768_345")
    assert stage_b[2] == 50
    assert stage_b[8] == "euler"
    assert stage_b[9] == "simple"
    assert stage_b[11] == "reference"
    assert stage_b[12] == "bf16"
    assert stage_b[13] is False


def test_benchmark_sampling_node_is_registered_and_describes_sampler_outputs():
    schema = KireiBenchmarkSampling.INPUT_TYPES()
    assert "model" in schema["required"]
    assert "scenario_id" in schema["required"]
    assert KireiBenchmarkSampling.RETURN_TYPES[:2] == ("SAMPLER", "SIGMAS")


def test_benchmark_start_requires_verified_recipe_token():
    start = KireiBenchmarkStart()
    try:
        start.start(_cpu_model(), {"verified": False})
    except TypeError as exc:
        assert "verified" in str(exc)
    else:
        raise AssertionError("unverified benchmark recipe must be rejected")


def test_benchmark_nodes_emit_verified_measurement_json_on_cpu():
    model = _cpu_model()
    payload, scenario, recipe = _resolve_scenario("native20_608x352_121")
    plan = _sampling_plan(payload, scenario, recipe)
    start = KireiBenchmarkStart()
    end = KireiBenchmarkEnd()

    passthrough, token = start.start(model, plan, reset_peak_vram=True)
    assert passthrough is model
    assert token["scenario_id"] == "native20_608x352_121"
    assert token["device"] == "cpu"
    assert token["scenario_schema_version"] >= 5
    assert token["sampling_plan"]["sampler_name"] == "res_multistep"
    assert token["recipe_spec"]["steps"] == 20

    response = end.end(token, after=object(), model=None, run_kind="warm")
    text = response["result"][0]
    output = json.loads(text)
    assert response["ui"]["text"] == [text]
    assert output["scenario_id"] == "native20_608x352_121"
    assert output["sampling_plan"]["verified"] is True
    assert output["sampling_plan"]["scheduler_name"] == "simple"
    assert output["run_kind"] == "warm"
    assert output["sampler_seconds"] >= 0.0
    assert output["peak_vram_bytes"] is None


def test_benchmark_nodes_reject_unknown_scenario_token():
    start = KireiBenchmarkStart()
    token = {"verified": True, "scenario_id": "does-not-exist"}
    try:
        start.start(_cpu_model(), token)
    except ValueError as exc:
        assert "unknown benchmark scenario" in str(exc)
    else:
        raise AssertionError("unknown scenario must be rejected")


def test_benchmark_nodes_always_invalidate_comfy_cache():
    assert math.isnan(KireiBenchmarkStart.IS_CHANGED())
    assert math.isnan(KireiBenchmarkEnd.IS_CHANGED())
