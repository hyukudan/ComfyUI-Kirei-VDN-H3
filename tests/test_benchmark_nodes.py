import json
import math
from types import SimpleNamespace

from vdn_h3.benchmark_nodes import KireiBenchmarkEnd, KireiBenchmarkStart


def test_benchmark_nodes_emit_measurement_json_on_cpu():
    model = SimpleNamespace(load_device="cpu")
    start = KireiBenchmarkStart()
    end = KireiBenchmarkEnd()

    passthrough, token = start.start(model, "scenario", reset_peak_vram=True)
    assert passthrough is model
    assert token["scenario_id"] == "scenario"
    assert token["device"] == "cpu"

    response = end.end(token, after=object(), model=None, run_kind="warm")
    text = response["result"][0]
    payload = json.loads(text)
    assert response["ui"]["text"] == [text]
    assert payload["scenario_id"] == "scenario"
    assert payload["run_kind"] == "warm"
    assert payload["sampler_seconds"] >= 0.0
    assert payload["peak_vram_bytes"] is None


def test_benchmark_nodes_always_invalidate_comfy_cache():
    assert math.isnan(KireiBenchmarkStart.IS_CHANGED())
    assert math.isnan(KireiBenchmarkEnd.IS_CHANGED())
