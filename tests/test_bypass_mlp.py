import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from vdn_h3.adapters import FactorPatch
from vdn_h3.bypass import (
    WeightedFactor,
    fc2_base_output,
    install_bypass,
    partition_factors,
    requires_weight_merge,
    swiglu,
)


class MLP(nn.Module):
    """Native MiniMax-H3 MLP shape: fc1 -> [gate; value] -> SwiGLU -> fc2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 12, bias=False)
        self.fc2 = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class Patcher:
    def __init__(self):
        self.model = nn.Module()
        self.model.diffusion_model = nn.Module()
        block = nn.Module()
        block.mlp = MLP()
        self.model.diffusion_model.blocks = nn.ModuleList([block])
        self.injections = {}

    def model_dtype(self):
        return torch.float32

    def get_model_object(self, key):
        value = self.model
        for part in key.split("."):
            value = getattr(value, part)
        return value

    def set_injections(self, key, injections):
        self.injections[key] = injections


def _fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    extension = types.ModuleType("comfy.patcher_extension")

    class PatcherInjection:
        def __init__(self, inject, eject):
            self.inject, self.eject = inject, eject

    extension.PatcherInjection = PatcherInjection
    comfy.patcher_extension = extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    monkeypatch.delitem(sys.modules, "comfy.model_patcher", raising=False)
    monkeypatch.delitem(sys.modules, "comfy.ops", raising=False)


def _fc2_patch(up, down, scale):
    return FactorPatch(
        key="diffusion_model.blocks.0.mlp.fc2.weight",
        up=up, down=down, offset=None, source="ff.net.2", scale=scale,
    )


def test_fc2_factors_are_never_forced_into_a_weight_merge():
    assert not requires_weight_merge("diffusion_model.blocks.0.mlp.fc2.weight")
    patch = _fc2_patch(torch.zeros(4, 2), torch.zeros(2, 6), 1.0)
    bypass, merge, curve = partition_factors([WeightedFactor(patch, 1.0)], "bypass")
    assert len(bypass) == 1 and not merge and not curve
    bypass, merge, curve = partition_factors([WeightedFactor(patch, 1.0)], "merge")
    assert not bypass and len(merge) == 1 and not curve


def test_mlp_hook_adds_the_exact_low_rank_term_from_the_swiglu_activation(monkeypatch):
    _fake_comfy(monkeypatch)
    torch.manual_seed(5)
    patcher = Patcher()
    mlp = patcher.get_model_object("diffusion_model.blocks.0.mlp")
    up, down, scale, strength = torch.randn(4, 2), torch.randn(2, 6), 0.5, 0.8
    count, runtime = install_bypass(patcher, [WeightedFactor(_fc2_patch(up, down, scale), strength)])
    assert count == 1 and runtime is not None

    x = torch.randn(3, 4)
    base = mlp(x).clone()
    injection = patcher.injections["vdn_h3_lora_bypass"][0]
    injection.inject(patcher)
    got = mlp(x)
    activation = swiglu(mlp.fc1(x))
    # Factors are stored in BF16 (the H3 compute dtype) with the scale folded into B.
    down_stored = down.to(torch.bfloat16).float()
    up_stored = (up.to(torch.bfloat16) * (scale * strength)).float()
    expected = base + F.linear(F.linear(activation, down_stored), up_stored)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    injection.eject(patcher)
    torch.testing.assert_close(mlp(x), base)


def test_fc2_base_output_keeps_the_native_fused_path_for_int8_weights(monkeypatch):
    calls = []
    ops = types.ModuleType("comfy.ops")

    def linear_input_act(linear, hidden, act):
        calls.append(act)
        return torch.full((hidden.shape[0], 4), 2.0)

    ops.linear_input_act = linear_input_act
    comfy = types.ModuleType("comfy")
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)

    int8_fc2 = types.SimpleNamespace(weight=types.SimpleNamespace(_layout_cls="TensorWiseINT8Layout"))
    hidden = torch.randn(3, 12)
    out = fc2_base_output(int8_fc2, hidden, swiglu(hidden))
    assert calls == ["swiglu"] and torch.equal(out, torch.full((3, 4), 2.0))

    plain = nn.Linear(6, 4, bias=False)
    torch.testing.assert_close(fc2_base_output(plain, hidden, swiglu(hidden)), plain(swiglu(hidden)))
    assert calls == ["swiglu"]
