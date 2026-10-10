import sys
import types

import pytest
import torch
import torch.nn as nn

from vdn_h3.adapters import FactorPatch
from vdn_h3.bypass import WeightedFactor, install_bypass


class ControlledLinear(nn.Module):
    def __init__(self, label, events):
        super().__init__()
        self.label = label
        self.events = events
        self.fail_next_hook = False

    def register_forward_hook(self, hook, **kwargs):
        self.events.append(f"{self.label}:hook")
        if self.fail_next_hook:
            self.fail_next_hook = False
            raise RuntimeError(f"blocked hook registration on {self.label}")
        handle = super().register_forward_hook(hook, **kwargs)
        original_remove = handle.remove

        def remove():
            self.events.append(f"{self.label}:base")
            original_remove()

        handle.remove = remove
        return handle

    def forward(self, x):
        return x.clone()


class SharedModel(nn.Module):
    def __init__(self, names=("proj",)):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.events = []
        for name in names:
            setattr(self.diffusion_model, name, ControlledLinear(name, self.events))


class Patcher:
    def __init__(self, shared_model):
        self.model = shared_model
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


def _factor(name, strength):
    patch = FactorPatch(
        key=f"diffusion_model.{name}.weight",
        up=torch.ones(2, 1),
        down=torch.ones(1, 2),
        offset=None,
        source=name,
        scale=1.0,
    )
    return WeightedFactor(patch, strength)


def _install(patcher, factors):
    install_bypass(patcher, factors)
    return patcher.injections["vdn_h3_lora_bypass"][0]


def test_shared_clone_reinjection_has_one_owner_and_preserves_external_hooks(monkeypatch):
    _fake_comfy(monkeypatch)
    shared = SharedModel()
    target = shared.diffusion_model.proj
    native_forward = target.forward

    def external_before(x):
        return native_forward(x) + 10

    target.forward = external_before
    clone_a, clone_b = Patcher(shared), Patcher(shared)
    injection_a = _install(clone_a, [_factor("proj", 1.0)])
    injection_b = _install(clone_b, [_factor("proj", 2.0)])
    x = torch.ones(1, 2)

    injection_a.inject(clone_a)
    torch.testing.assert_close(target(x), torch.full_like(x, 13))
    injection_a.inject(clone_a)  # repeated injection must not double the delta
    torch.testing.assert_close(target(x), torch.full_like(x, 13))

    injection_b.inject(clone_b)
    torch.testing.assert_close(target(x), torch.full_like(x, 15))
    injection_a.eject(clone_a)  # stale clone A must not eject active clone B
    torch.testing.assert_close(target(x), torch.full_like(x, 15))

    injection_a.inject(clone_a)
    torch.testing.assert_close(target(x), torch.full_like(x, 13))
    injection_b.eject(clone_b)  # stale clone B must not eject active clone A
    torch.testing.assert_close(target(x), torch.full_like(x, 13))
    injection_a.eject(clone_a)
    assert target.forward is external_before

    # A later forward replacement remains installed when our independent post-hook
    # is removed, and no stale VDN wrapper remains in its call chain.
    injection_a.inject(clone_a)
    vdn_forward = target.forward

    def external_after(x):
        return vdn_forward(x) + 20

    target.forward = external_after
    torch.testing.assert_close(target(x), torch.full_like(x, 33))
    injection_a.eject(clone_a)
    assert target.forward is external_after
    torch.testing.assert_close(target(x), torch.full_like(x, 31))
    target.forward = vdn_forward
    injection_a.eject(clone_a)
    assert target.forward is external_before


def test_failed_takeover_rolls_back_in_lifo_order_and_restores_owner(monkeypatch):
    _fake_comfy(monkeypatch)
    shared = SharedModel(("first", "second"))
    clone_a, clone_b = Patcher(shared), Patcher(shared)
    factors_a = [_factor("first", 1.0), _factor("second", 1.0)]
    factors_b = [_factor("first", 3.0), _factor("second", 3.0)]
    injection_a = _install(clone_a, factors_a)
    injection_b = _install(clone_b, factors_b)
    injection_a.inject(clone_a)

    shared.events.clear()
    shared.diffusion_model.second.fail_next_hook = True
    with pytest.raises(RuntimeError, match="blocked hook registration"):
        injection_b.inject(clone_b)

    assert shared.events == [
        "second:base",
        "first:base",
        "first:hook",
        "second:hook",
        "first:base",
        "first:hook",
        "second:hook",
    ]
    x = torch.ones(1, 2)
    torch.testing.assert_close(shared.diffusion_model.first(x), torch.full_like(x, 3))
    torch.testing.assert_close(shared.diffusion_model.second(x), torch.full_like(x, 3))

    injection_b.eject(clone_b)  # failed/stale owner cannot affect restored clone A
    torch.testing.assert_close(shared.diffusion_model.first(x), torch.full_like(x, 3))
    injection_a.eject(clone_a)
    torch.testing.assert_close(shared.diffusion_model.first(x), x)
    torch.testing.assert_close(shared.diffusion_model.second(x), x)


def test_merge_only_clone_displaces_live_bypass_owner(monkeypatch):
    _fake_comfy(monkeypatch)
    shared = SharedModel()
    bypass_clone, merge_clone = Patcher(shared), Patcher(shared)
    bypass_injection = _install(bypass_clone, [_factor("proj", 2.0)])
    x = torch.ones(1, 2)

    bypass_injection.inject(bypass_clone)
    torch.testing.assert_close(shared.diffusion_model.proj(x), torch.full_like(x, 5))

    count, runtime = install_bypass(merge_clone, [])
    assert count == 0 and runtime is None
    merge_injection = merge_clone.injections["vdn_h3_lora_bypass"][0]
    merge_injection.inject(merge_clone)
    torch.testing.assert_close(shared.diffusion_model.proj(x), x)

    bypass_injection.eject(bypass_clone)  # stale bypass clone cannot undo merge ownership
    torch.testing.assert_close(shared.diffusion_model.proj(x), x)
    merge_injection.eject(merge_clone)
