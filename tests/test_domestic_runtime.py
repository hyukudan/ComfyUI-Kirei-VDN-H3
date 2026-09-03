import json

import torch

from vdn_h3 import window
from vdn_h3.bypass import CompositeQKVBypassAdapter, FrugalLoRABypassAdapter
from vdn_h3.calibration import CALIBRATION_VERSION, CalibrationStore, calibration_signature
from vdn_h3.runtime import OptimizedLinearBranch, SharedBranchRuntime
from vdn_h3.weights import (
    FP8_STREAMED_PROJECTION_KEY,
    STREAMED_PROJECTION_KEY,
    ManagedBranchWeights,
)


def _weights(hidden=7, heads=2, dim=3, *, short=False):
    channels = heads * dim
    value = {
        "to_out_linear.weight": torch.randn(hidden, channels) * 0.1,
        "beta_proj.weight": torch.randn(heads, hidden) * 0.1,
        "alpha.down.weight": torch.randn(dim, hidden) * 0.1,
        "alpha.up.weight": torch.randn(channels, dim) * 0.1,
        "alpha.dt_bias": torch.randn(channels) * 0.1,
        "alpha.A_log": torch.zeros(heads),
        "output_gate.down.weight": torch.randn(dim, hidden) * 0.1,
        "output_gate.up.weight": torch.randn(channels, dim) * 0.1,
        "output_gate.up.bias": torch.randn(channels) * 0.1,
        "norm.weight": torch.ones(dim),
    }
    if short:
        value.update(
            {
                "short_conv.k_sp.weight": torch.randn(channels, 1, 3, 3) * 0.05,
                "short_conv.k_tm.weight": torch.randn(channels, 1, 5) * 0.05,
                "short_conv.v_sp.weight": torch.randn(channels, 1, 3, 3) * 0.05,
                "short_conv.v_tm.weight": torch.randn(channels, 1, 5) * 0.05,
            }
        )
    return value


def test_tiled_projected_delta_matches_untiled_with_temporal_halo():
    torch.manual_seed(901)
    frames, per_frame, heads, dim, hidden = 7, 4, 2, 3, 7
    rows = frames * per_frame
    weights = _weights(hidden, heads, dim, short=True)
    bounds = window.window_bounds(frames, 1, 2)
    x = torch.randn(rows, hidden)
    q = torch.randn(rows, heads, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    shared = SharedBranchRuntime(kernel_backend="eager", compile_policy="off", tile_frames=2)
    tiled = OptimizedLinearBranch(
        weights,
        heads,
        dim,
        short_conv=("k", "v"),
        enable_text_state=False,
        kernel_backend="eager",
        compile_policy="off",
        tile_frames=2,
    ).set_runtime(runtime_cache=shared, diagnostics=None)
    untiled = OptimizedLinearBranch(
        weights,
        heads,
        dim,
        short_conv=("k", "v"),
        enable_text_state=False,
        kernel_backend="eager",
        compile_policy="off",
        tile_frames=0,
    ).set_runtime(
        runtime_cache=SharedBranchRuntime(
            kernel_backend="eager", compile_policy="off", tile_frames=0
        ),
        diagnostics=None,
    )

    with torch.no_grad():
        got = tiled.projected_delta(
            weights,
            x,
            q,
            k,
            v,
            frames,
            per_frame,
            bounds,
            frame_size=(2, 2),
            skip_ends=True,
            inference=True,
        )
        want = untiled.projected_delta(
            weights,
            x,
            q,
            k,
            v,
            frames,
            per_frame,
            bounds,
            frame_size=(2, 2),
            skip_ends=True,
            inference=True,
        )
    torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-4)
    assert torch.count_nonzero(got[:per_frame]) == 0
    assert torch.count_nonzero(got[-per_frame:]) == 0


def test_shared_runtime_is_reused_by_multiple_branches():
    weights = _weights(short=False)
    runtime = SharedBranchRuntime(kernel_backend="eager", compile_policy="off")
    first = OptimizedLinearBranch(weights, 2, 3, short_conv=(), enable_text_state=False)
    second = OptimizedLinearBranch(weights, 2, 3, short_conv=(), enable_text_state=False)
    first.set_runtime(runtime_cache=runtime)
    second.set_runtime(runtime_cache=runtime)
    assert first._gather_cache is second._gather_cache is runtime.gather
    assert first._kernel_cache is second._kernel_cache is runtime.kernels
    assert first._compiler_cache is second._compiler_cache is runtime.compiler


def test_lora_composite_fuses_up_gemms_per_output_slice_exactly():
    torch.manual_seed(902)
    x = torch.randn(5, 4)
    base = torch.zeros(5, 12)
    q1 = FrugalLoRABypassAdapter(torch.randn(4, 2), torch.randn(2, 4), 0.5)
    q2 = FrugalLoRABypassAdapter(torch.randn(4, 1), torch.randn(1, 4), -0.25)
    k1 = FrugalLoRABypassAdapter(torch.randn(4, 3), torch.randn(3, 4), 0.75)
    expected = base.clone()
    expected[:, :4] += q1.delta(x) + q2.delta(x)
    expected[:, 4:8] += k1.delta(x)

    composite = CompositeQKVBypassAdapter(
        [(q1, (0, 4)), (q2, (0, 4)), (k1, (4, 4))]
    )
    got = composite.apply(x, base.clone())
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)
    assert len(composite.group_ups) == 2


def test_hybrid_store_keeps_small_weights_resident_and_projection_streamed_on_cpu():
    block = {
        "to_out_linear.weight": torch.ones(8, 8),
        "alpha.A_log": torch.ones(2),
        "small": torch.ones(4),
    }
    store = ManagedBranchWeights([block], mode="hybrid", pin_strategy="none")
    assert set(store.model.resident.block(0).keys()) == {"alpha.A_log", "small"}
    assert set(store.model.streamed.block(0).keys()) == {"to_out_linear.weight"}
    got = store.weights_on(0, "cpu", torch.bfloat16)
    assert got["alpha.A_log"].dtype == torch.float32
    assert got["small"].dtype == torch.bfloat16
    assert got["to_out_linear.weight"].dtype == torch.bfloat16
    telemetry = store.telemetry()
    assert telemetry["resident_bytes"] > 0 and telemetry["streamed_bytes"] > 0


def test_fp8_storage_dtype_is_preserved_by_hybrid_store():
    fp8 = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
    block = {
        FP8_STREAMED_PROJECTION_KEY: fp8,
        "to_out_linear.weight_scale": torch.ones(1, 8, dtype=torch.float32),
        "small": torch.ones(4, dtype=torch.bfloat16),
    }
    store = ManagedBranchWeights(
        [block],
        mode="hybrid",
        pin_strategy="none",
        streamed_keys=(FP8_STREAMED_PROJECTION_KEY,),
    )
    got = store.weights_on(0, "cpu", torch.bfloat16)
    assert got[FP8_STREAMED_PROJECTION_KEY].dtype == torch.float8_e4m3fn
    assert got["to_out_linear.weight_scale"].dtype == torch.float32
    assert got["small"].dtype == torch.bfloat16


def test_hybrid_store_accepts_bf16_edges_and_fp8_interior():
    fp8 = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
    blocks = [
        {STREAMED_PROJECTION_KEY: torch.ones(8, 8), "small": torch.ones(1)},
        {
            FP8_STREAMED_PROJECTION_KEY: fp8,
            "to_out_linear.weight_scale": torch.ones(1, 8),
            "small": torch.ones(1),
        },
        {STREAMED_PROJECTION_KEY: torch.ones(8, 8), "small": torch.ones(1)},
    ]
    store = ManagedBranchWeights(
        blocks,
        mode="hybrid",
        pin_strategy="none",
        streamed_keys=(STREAMED_PROJECTION_KEY, FP8_STREAMED_PROJECTION_KEY),
    )
    assert set(store.model.streamed.block(0).keys()) == {STREAMED_PROJECTION_KEY}
    assert set(store.model.streamed.block(1).keys()) == {FP8_STREAMED_PROJECTION_KEY}
    assert set(store.model.streamed.block(2).keys()) == {STREAMED_PROJECTION_KEY}


def test_calibration_store_roundtrip_and_geometry_signature(tmp_path):
    path = tmp_path / "calibration.json"
    store = CalibrationStore(path)
    store.record(
        "signature",
        winner="grouped",
        results={"grouped": {"ms": 1.0, "allclose": True}},
    )
    store.save()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["version"] == CALIBRATION_VERSION
    again = CalibrationStore(path)
    assert again.lookup("signature") == "grouped"
    assert again.snapshot()["entries"] == 1

    q = torch.zeros(20, 2, 4)
    bounds = window.window_bounds(3, 1)
    first = calibration_signature(
        q, 3, bounds, "both", groups=1,
        video_start=2, video_end=14, tokens_per_frame=4,
    )
    second = calibration_signature(
        q, 3, bounds, "both", groups=1,
        video_start=5, video_end=14, tokens_per_frame=3,
    )
    assert first != second


def test_calibration_signature_tracks_node_version_and_backend_inventory(monkeypatch):
    from vdn_h3 import __version__, calibration

    calibration.runtime_environment.cache_clear()
    env = calibration.runtime_environment()
    assert env["node"] == __version__
    assert env["torch"] == torch.__version__
    assert "grouped" in env["backends"]
    q = torch.zeros(20, 2, 4)
    bounds = window.window_bounds(3, 1)
    geometry = dict(groups=1, video_start=2, video_end=14, tokens_per_frame=4)
    before = calibration_signature(q, 3, bounds, "both", **geometry)
    parsed = json.loads(before)
    assert parsed["version"] == CALIBRATION_VERSION == 3
    assert parsed["env"]["node"] == __version__
    calibration.runtime_environment.cache_clear()
    monkeypatch.setattr(window, "flash2_available", lambda cache=None: True)
    after = calibration_signature(q, 3, bounds, "both", **geometry)
    assert after != before
    assert "flash2" in json.loads(after)["env"]["backends"]
    calibration.runtime_environment.cache_clear()


def test_calibration_store_ignores_previous_schema_versions(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": 2, "entries": {"sig": {"winner": "flex"}}}), encoding="utf-8")
    store = CalibrationStore(path)
    assert store.lookup("sig") is None
    assert store.snapshot()["entries"] == 0
