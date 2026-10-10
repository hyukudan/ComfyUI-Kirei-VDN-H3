"""Reduced CPU oracles executing pinned OpenVDN source with synthetic weights.

Adapted from xmarre/ComfyUI-VDN-H3-Plus (Apache-2.0), copyright 2026
Saganaki22 and contributors. OpenVDN implementation copyright 2026 the VDN authors.
See LICENSE, NOTICE and THIRD_PARTY.md. No model weights are downloaded or loaded.

Only eager FP32 window/linear/hybrid math and scheduler grids are compared. CUDA,
RoPE, FP8, trained checkpoint behavior and end-to-end video parity are excluded.
Set OPENVDN_ROOT to a clean checkout of OPENVDN_SHA; absent roots skip the suite.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
from torch import nn
import yaml

from vdn_h3 import branch as port_branch
from vdn_h3 import window as port_window
from vdn_h3.branch import LinearBranch
from vdn_h3.hybrid import VDNState, make_vdn_forward
from vdn_h3.kernels import activate
from vdn_h3.layout import VDNLayout, publish_layout

OPENVDN_SHA = "b8cb28fbfca0266d1c7742a9f25ab8b58191de97"
ROOT = Path(os.environ.get("OPENVDN_ROOT", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENVDN_ROOT"), reason="OPENVDN_ROOT checkout is required for official oracle")


def _pkg(name):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    return mod


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_official_hybrid():
    for name in (
        "src", "src.models", "src.models.ops", "src.models.linear_attention",
        "src.models.softmax_attention", "src.checkpoints",
        "diffusers", "diffusers.models", "diffusers.models.transformers",
    ):
        _pkg(name)

    # OpenVDN's softmax kernels import the diffusers MiniMax rotary helper at module
    # import time. This CPU oracle never supplies rotary_emb, so install the inert
    # dependency before loading those kernels rather than depending on diffusers.
    minimax = types.ModuleType("diffusers.models.transformers.transformer_minimax_h3")
    minimax._apply_rotary_emb = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("RoPE helper must not run when rotary_emb=None"))
    sys.modules[minimax.__name__] = minimax

    sequence = _load("src.models.sequence_layout", "src/models/sequence_layout.py")
    window = _load("official_hybrid_window", "src/models/softmax_attention/window.py")
    delta = _load("src.models.linear_attention.delta_rule",
                  "src/models/linear_attention/delta_rule.py")
    scan = _load("src.models.linear_attention.scan", "src/models/linear_attention/scan.py")

    temporal = types.ModuleType("src.models.ops.temporal_conv")
    temporal.temporal_conv_activate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("CUDA temporal kernel must not run in CPU hybrid oracle"))
    sys.modules[temporal.__name__] = temporal
    features = _load("src.models.linear_attention.features",
                     "src/models/linear_attention/features.py")
    _load("src.models.ops.rms_norm", "src/models/ops/rms_norm.py")
    gates = _load("src.models.attention_gates", "src/models/attention_gates.py")
    _load("src.models.linear_attention.kernels",
          "src/models/linear_attention/kernels.py")
    _load("src.models.linear_attention.layers", "src/models/linear_attention/layers.py")

    key_mapping = types.ModuleType("src.checkpoints.key_mapping")
    key_mapping.SHORT_CONV_TARGETS = ("q", "k", "v")
    key_mapping.ANCHOR_FRAME_MODES = ("none", "rows", "columns", "both")
    sys.modules[key_mapping.__name__] = key_mapping

    linear_pkg = sys.modules["src.models.linear_attention"]
    linear_pkg.DELTA_BACKENDS = delta.DELTA_BACKENDS
    branch = _load("official_hybrid_branch", "src/models/linear_attention/branch.py")
    linear_pkg.BidirectionalLinearBranch = branch.BidirectionalLinearBranch

    # Register under the exact canonical module name imported by the released
    # HybridAttention class. This keeps the oracle's dependency wiring faithful to
    # OpenVDN rather than substituting a lookalike helper module.
    softmax_kernels = _load(
        "src.models.softmax_attention.kernels",
        "src/models/softmax_attention/kernels.py",
    )
    softmax_pkg = sys.modules["src.models.softmax_attention"]
    softmax_pkg.apply_softmax_gate = softmax_kernels.apply_softmax_gate
    softmax_pkg.window_bounds = window.window_bounds
    softmax_pkg.window_softmax_reference = window.window_softmax_reference
    softmax_pkg.build_window_block_mask = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("FlexAttention mask must not run in CPU reference oracle"))
    softmax_pkg.window_softmax_flex = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("FlexAttention must not run in CPU reference oracle"))

    # HybridAttention's full-cover dense dispatch is deliberately unreachable in this
    # windowed reduced case. Keep it as a fail-fast stub so the oracle cannot silently
    # switch to a different attention implementation.
    attention_dispatch = types.ModuleType("diffusers.models.attention_dispatch")
    attention_dispatch.dispatch_attention_fn = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dense diffusers dispatch must not run in windowed oracle"))
    sys.modules[attention_dispatch.__name__] = attention_dispatch

    fp8 = types.ModuleType("src.models.ops.fp8_linear")

    class Fp8Linear(nn.Module):
        pass

    fp8.Fp8Linear = Fp8Linear
    fp8.quantize_activation = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("FP8 path must not run in CPU oracle"))
    sys.modules[fp8.__name__] = fp8

    hybrid = _load("official_vdn_hybrid_attention", "src/models/hybrid_attention.py")
    for name in ("diffusers.modular_pipelines", "diffusers.modular_pipelines.minimax_h3"):
        _pkg(name)
    packing = types.ModuleType("diffusers.modular_pipelines.minimax_h3.before_denoise")
    packing.MiniMaxH3PrepareLayoutStep = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("latent packing must not run in scheduler oracle"))
    packing.patchify_video_latents = packing.MiniMaxH3PrepareLayoutStep
    sys.modules[packing.__name__] = packing
    timesteps = _load("official_vdn_t2va_batch", "src/training/t2va_batch.py")
    return types.SimpleNamespace(
        sequence=sequence, window=window, delta=delta, scan=scan,
        features=features, gates=gates, branch=branch,
        hybrid=hybrid, timesteps=timesteps)


@pytest.fixture(scope="module")
def official():
    assert ROOT.is_dir(), f"OPENVDN_ROOT does not exist: {ROOT}"
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert sha == OPENVDN_SHA, f"official oracle requires {OPENVDN_SHA}, got {sha}"
    subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", "HEAD", "--", "src", "configs"], check=True)
    prefixes = ("src", "diffusers", "official_")
    saved = {name: module for name, module in sys.modules.items()
             if name.startswith(prefixes)}
    for name in saved:
        del sys.modules[name]
    try:
        yield _load_official_hybrid()
    finally:
        for name in list(sys.modules):
            if name.startswith(prefixes):
                del sys.modules[name]
        sys.modules.update(saved)


def test_window_bounds_frame_and_chunk_modes(official):
    for frames in (1, 2, 7, 23):
        for radius in (0, 1, 3):
            for chunk in (0, 1, 5, 8):
                got = port_window.window_bounds(frames, radius, chunk)
                want = official.window.window_bounds(frames, radius, chunk)
                assert got == want
                assert port_window.full_coverage(got, frames) == all(
                    lo <= 0 and hi >= frames - 1 for lo, hi in want)


def test_window_softmax_all_anchor_modes_against_official(official):
    torch.manual_seed(100)
    frames, per_frame, heads, dim = 9, 3, 2, 8
    video_start = 4
    total = video_start + frames * per_frame + 5
    layout = official.sequence.SequenceLayout(
        seq_len=total, video_start=video_start, num_frames=frames,
        tokens_per_frame=per_frame)
    q, k, v = [torch.randn(total, heads, dim) for _ in range(3)]
    bounds = official.window.window_bounds(frames, 1, 3)
    scale = dim ** -0.5
    for anchor in ("none", "rows", "columns", "both"):
        want = official.window.window_softmax_reference(
            q, k, v, layout, bounds, scale, anchor_frames=anchor)
        got = port_window.window_softmax_grouped(
            q, k, v, layout.video_start, layout.video_end,
            frames, per_frame, bounds, scale, anchor_frames=anchor)
        assert torch.allclose(got, want, atol=2e-6, rtol=2e-6), anchor


def _spd_stats(frames=5, heads=2, dim=8):
    torch.manual_seed(101)
    alpha = torch.rand(frames, heads, dim) * 0.4 + 0.55
    x = torch.randn(frames, heads, dim, dim) * 0.02
    a = x @ x.transpose(-1, -2)
    b = torch.randn(frames, heads, dim, dim) * 0.1
    return alpha, a, b


@pytest.mark.parametrize("rule", ["vdn_solve", "sana_scaled"])
def test_delta_rules_direct_official(rule, official):
    alpha, a, b = _spd_stats()
    tokens = 17
    port_cls = port_branch.DELTA_BACKENDS[rule]
    off_cls = official.delta.DELTA_BACKENDS[rule]
    got_t, got_i = port_cls(tokens).factor_apply(alpha, a, b)
    want_t, want_i = off_cls(tokens).factor_apply(alpha, a, b)
    assert torch.allclose(got_t, want_t, atol=2e-6, rtol=2e-6)
    assert torch.allclose(got_i, want_i, atol=2e-6, rtol=2e-6)


def test_frame_statistics_direct_official(official):
    torch.manual_seed(102)
    frames, heads, rows, dim = 4, 3, 7, 8
    k = torch.randn(frames, heads, rows, dim)
    v = torch.randn(frames, heads, rows, dim)
    beta = torch.sigmoid(torch.randn(frames, heads, rows))
    got_a, got_b = port_branch.frame_statistics(k, v, beta, a_fp32=True)
    want_a, want_b = official.scan.frame_statistics(k, v, beta, a_fp32=True, inference=False)
    assert torch.allclose(got_a, want_a, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got_b, want_b, atol=1e-6, rtol=1e-6)


def test_forward_reverse_scans_and_alpha_bridge_direct_official(official):
    alpha, a, b = _spd_stats(frames=7)
    text = torch.randn(2, 8, 8) * 0.03
    port_backend = port_branch.VdnDelta(11)
    official_backend = official.delta.VdnDelta(11)
    got_prefix, got_suffix = port_branch.run_scans(
        port_backend, alpha, a, b, text_state=text)
    want_prefix, want_suffix = official.scan._run_scans_inference(
        official_backend, alpha, a, b, text_state=text)
    assert torch.allclose(got_prefix, want_prefix, atol=2e-6, rtol=2e-6)
    assert torch.allclose(got_suffix, want_suffix, atol=2e-6, rtol=2e-6)

    bounds = official.window.window_bounds(7, 1, 3)
    got = port_branch.gather_linear_state(
        got_prefix, got_suffix, alpha, bounds, bridge="alpha", text_state=text)
    want = official.scan.gather_linear_state(
        want_prefix, want_suffix, alpha, bounds, bridge="alpha", text_state=text,
        inference=False)
    assert torch.allclose(got, want, atol=3e-6, rtol=3e-6)


def test_activation_and_short_conv_feature_path_direct_official(official):
    torch.manual_seed(103)
    frames, gh, gw, heads, dim = 3, 2, 3, 2, 4
    rows = frames * gh * gw
    tokens = torch.randn(rows, heads, dim)
    channels = heads * dim

    for l2norm in (False, True):
        got = activate(tokens, l2norm)
        want = official.features._activate_body(tokens, l2norm)
        assert torch.allclose(got, want, atol=1e-7, rtol=1e-7)

    module = official.features.LinearAttentionSepConv(channels, ("k", "v"))
    sp = torch.randn(channels, 1, 5, 5) * 0.05
    tm = torch.randn(channels, 1, 5) * 0.05
    with torch.no_grad():
        module.k_sp.weight.copy_(sp)
        module.k_tm.weight.copy_(tm)
    want = official.features.prepare_linear_features(
        tokens, True, conv=module, proj="k", num_frames=frames,
        frame_size=(gh, gw))
    got = port_branch.conv_features(
        tokens, sp, tm, frames, (gh, gw), l2norm=True)
    assert torch.allclose(got, want, atol=2e-6, rtol=2e-6)


def _branch_weights(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


@pytest.mark.parametrize("short_conv", [(), ("k", "v")])
def test_complete_bidirectional_linear_branch_direct_official(short_conv, official):
    torch.manual_seed(104 if not short_conv else 105)
    hidden, heads, dim = 16, 2, 4
    frames, gh, gw = 5, 2, 2
    per_frame = gh * gw
    rows = frames * per_frame
    text_len = 6

    off = official.branch.BidirectionalLinearBranch(
        hidden, heads, dim, delta_rule="vdn_solve", bridge="alpha",
        a_fp32=True, short_conv=short_conv)
    weights = _branch_weights(off)
    port = port_branch.LinearBranch(
        weights, heads, dim, delta_rule="vdn_solve", bridge="alpha",
        a_fp32=True, short_conv=short_conv, enable_text_state=True,
        linear_kernels="eager")

    xv = torch.randn(rows, hidden)
    qkv = tuple(torch.randn(rows, heads, dim) for _ in range(3))
    text_x = torch.randn(text_len, hidden)
    text_qkv = tuple(torch.randn(text_len, heads, dim) for _ in range(3))
    bounds = official.window.window_bounds(frames, 1, 2)

    want = off(
        xv, frames, per_frame, bounds, qkv,
        frame_size=(gh, gw) if short_conv else None,
        skip_ends=True, text_x=text_x, text_qkv_raw=text_qkv, inference=False)
    got = port.readout(
        weights, xv, *qkv, frames, per_frame, bounds,
        frame_size=(gh, gw) if short_conv else None,
        text_x=text_x, text_k_raw=text_qkv[1], text_v_raw=text_qkv[2],
        skip_ends=True)
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5)


def test_released_stage_dmd_configuration_is_eight_step(official):
    # Directly inspect the official repository artifact rather than duplicating this
    # release fact in a local fixture.
    cfg = yaml.safe_load((ROOT / "configs/training/stage_dmd_vdn.yaml").read_text())
    assert cfg["turbo"]["num_steps"] == 8
    assert cfg["turbo"]["video_shift"] == 12.0
    assert cfg["turbo"]["audio_shift"] == 3.0


@pytest.mark.parametrize("steps", [8, 50])
def test_comfy_paired_timestep_grid_against_official(official, steps):
    # Kirei uses native Comfy simple scheduling; compare the real helper rather
    # than copying the upstream shift formula into a local expected tensor.
    from comfy.model_sampling import ModelSamplingAV
    from comfy.ldm.minimax.model import time_shift_sigma
    from vdn_h3.benchmark_nodes import _build_sampler_and_sigmas

    sampling = ModelSamplingAV()
    sampling.set_parameters(shift=12.0, audio_shift=3.0, multiplier=1.0)
    model = types.SimpleNamespace(model_sampling=sampling)
    _, sigmas = _build_sampler_and_sigmas(model, {
        "sampler_name": "euler", "scheduler_name": "simple", "steps": steps,
        "denoise": 1.0,
    })
    want_video, want_audio = official.timesteps.few_step_timesteps(steps, 12.0, 3.0)
    got_video = 1.0 - sigmas[:-1]
    got_audio = 1.0 - time_shift_sigma(sigmas[:-1], 12.0, 3.0)
    assert len(sigmas) == steps + 1 and sigmas[-1] == 0
    torch.testing.assert_close(got_video, want_video, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got_audio, want_audio, atol=2e-6, rtol=2e-6)


class _OfficialOrigAttention(nn.Module):
    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(hidden, inner, bias=False)
        self.to_k = nn.Linear(hidden, inner, bias=False)
        self.to_v = nn.Linear(hidden, inner, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.ModuleList([nn.Linear(inner, hidden, bias=False), nn.Identity()])
        self.processor = object()


class _PortAttention(nn.Module):
    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.out_proj = nn.Linear(inner, hidden, bias=False)


def _copy_base_attention(orig, port):
    with torch.no_grad():
        port.qkv_proj.weight.copy_(torch.cat(
            [orig.to_q.weight, orig.to_k.weight, orig.to_v.weight], dim=0))
        port.q_norm.weight.copy_(orig.norm_q.weight)
        port.k_norm.weight.copy_(orig.norm_k.weight)
        port.out_proj.weight.copy_(orig.to_out[0].weight)


@pytest.mark.parametrize("anchor", ["none", "rows", "columns", "both"])
@pytest.mark.parametrize("short_conv", [(), ("k", "v")])
def test_complete_hybrid_attention_direct_official(official, anchor, short_conv):
    torch.manual_seed(1701)
    hidden, heads, head_dim = 16, 2, 4
    text_len = 4
    frames, tokens_per_frame = 5, 3
    video_start = text_len
    video_end = video_start + frames * tokens_per_frame
    seq_len = video_end

    orig = _OfficialOrigAttention(hidden, heads, head_dim)
    off = official.hybrid.HybridAttention(
        orig,
        hidden_size=hidden,
        delta_rule="vdn_solve",
        radius=1,
        chunk=2,
        enable_softmax_gate=True,
        linear_head_dim=head_dim,
        softmax_impl="ref",
        anchor_frames=anchor,
        enable_text_state=True,
        bridge="alpha",
        a_fp32=True,
        short_conv=short_conv,
    )
    with torch.no_grad():
        off.softmax_gate.up.weight.normal_(std=0.1)
        off.softmax_gate.up.bias.normal_(std=0.1)
    off.layout = official.sequence.SequenceLayout(
        seq_len=seq_len,
        video_start=video_start,
        num_frames=frames,
        tokens_per_frame=tokens_per_frame,
        frame_height=1,
        frame_width=tokens_per_frame,
        text_start=0,
        text_len=text_len,
    )

    port_attn = _PortAttention(hidden, heads, head_dim)
    _copy_base_attention(orig, port_attn)

    weights = {
        name: tensor.detach().clone()
        for name, tensor in off.linear_attention.state_dict().items()
    }
    weights["to_out_linear.weight"] = off.to_out_linear.weight.detach().clone()
    weights["softmax_gate.up.weight"] = off.softmax_gate.up.weight.detach().clone()
    weights["softmax_gate.up.bias"] = off.softmax_gate.up.bias.detach().clone()

    port_linear = LinearBranch(
        weights,
        heads,
        head_dim,
        delta_rule="vdn_solve",
        bridge="alpha",
        a_fp32=True,
        short_conv=short_conv,
        enable_text_state=True,
        linear_kernels="eager",
    )
    cfg = {
        "radius": 1,
        "chunk": 2,
        "anchor_frames": anchor,
        "enable_softmax_gate": True,
        "linear_enabled": True,
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
        "a_fp32": True,
        "linear_head_dim": head_dim,
        "short_conv": short_conv,
        "enable_text_state": True,
    }
    state = VDNState("official-hybrid-oracle", cfg, [port_linear], heads, head_dim,
                     attention_backend="grouped", kernel_backend="eager", compile_policy="off")
    layout = VDNLayout(
        video_start=video_start,
        video_end=video_end,
        num_frames=frames,
        tokens_per_frame=tokens_per_frame,
        frame_size=(1, tokens_per_frame),
        text_start=0,
        text_len=text_len,
        seq_len=seq_len,
        bounds=tuple(port_window.window_bounds(frames, 1, 2)),
        full_cover=False,
        anchor_frames=anchor,
    )

    x = torch.randn(seq_len, hidden)
    with torch.no_grad():
        want = off(x.unsqueeze(0), rotary_emb=None, attention_mask=None)[0]
        with publish_layout(layout):
            got = make_vdn_forward(port_attn, state, 0)(
                x, rope_freqs=None, transformer_options={})

    assert got.shape == want.shape == (seq_len, hidden)
    assert torch.allclose(got, want, atol=3e-5, rtol=3e-5), (
        f"complete hybrid mismatch: max={(got - want).abs().max().item():.3e}")
