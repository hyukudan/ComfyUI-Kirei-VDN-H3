import json
import math
from types import SimpleNamespace

from vdn_h3.benchmark_nodes import (
    KireiBenchmarkEnd,
    KireiBenchmarkScenario,
    KireiBenchmarkStart,
)


def test_benchmark_scenario_exposes_connectable_recipe_values():
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
        profile,
        projection,
        apply_turbo,
        text,
    ) = values
    assert scenario_id == "vdn_dmd_bf16_8step_608x352_241"
    assert recipe_id == "vdn_stage_dmd_8"
    assert (steps, width, height, frames) == (8, 608, 352, 241)
    assert (shift_v, shift_a) == (12.0, 3.0)
    assert profile == "auto"
    assert projection == "bf16"
    assert apply_turbo is True
    payload = json.loads(text)
    assert payload["recipe"]["required_adapters"] == ["default", "turbo"]
    assert payload["runtime_controls"]["projection_precision"] == "bf16"


def test_benchmark_scenario_exposes_fp8_and_stage_b_controls():
    node = KireiBenchmarkScenario()
    fp8 = node.resolve("vdn_dmd_fp8_8step_608x352_241")
    assert fp8[8] == "auto"
    assert fp8[9] == "fp8"
    assert fp8[10] is True

    stage_b = node.resolve("vdn_stage_b_bf16_50step_1344x768_345")
    assert stage_b[2] == 50
    assert stage_b[8] == "reference"
    assert stage_b[9] == "bf16"
    assert stage_b[10] is False


def test_benchmark_nodes_emit_recipe_measurement_json_on_cpu():
    model = SimpleNamespace(
        load_device="cpu",
        model=SimpleNamespace(model_sampling=SimpleNamespace(shift=12.0, audio_shift=3.0)),
    )
    start = KireiBenchmarkStart()
    end = KireiBenchmarkEnd()

    passthrough, token = start.start(model, "native20_608x352_121", reset_peak_vram=True)
    assert passthrough is model
    assert token["scenario_id"] == "native20_608x352_121"
    assert token["device"] == "cpu"
    assert token["scenario_schema_version"] >= 4
    assert token["recipe_spec"]["steps"] == 20
    assert token["sampling"]["video_shift"] == 12.0
    assert token["sampling"]["audio_shift"] == 3.0

    response = end.end(token, after=object(), model=None, run_kind="warm")
    text = response["result"][0]
    payload = json.loads(text)
    assert response["ui"]["text"] == [text]
    assert payload["scenario_id"] == "native20_608x352_121"
    assert payload["scenario_spec"]["recipe_id"] == "native_standard_20"
    assert payload["recipe_spec"]["steps"] == 20
    assert payload["sampling"]["video_shift"] == 12.0
    assert payload["run_kind"] == "warm"
    assert payload["sampler_seconds"] >= 0.0
    assert payload["peak_vram_bytes"] is None


def test_benchmark_nodes_reject_inactive_or_unknown_scenarios():
    model = SimpleNamespace(load_device="cpu")
    start = KireiBenchmarkStart()
    try:
        start.start(model, "does-not-exist")
    except ValueError as exc:
        assert "unknown benchmark scenario" in str(exc)
    else:
        raise AssertionError("unknown scenario must be rejected")


def test_benchmark_nodes_always_invalidate_comfy_cache():
    assert math.isnan(KireiBenchmarkStart.IS_CHANGED())
    assert math.isnan(KireiBenchmarkEnd.IS_CHANGED())
