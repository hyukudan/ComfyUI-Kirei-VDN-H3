import copy

import pytest
import torch
import torch.nn as nn

from vdn_h3_private.apply import DeltaPatch, apply_delta_patches, clone_and_apply
from vdn_h3_private.hybrid import VDNState, apply_vdn


class TinyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.head_dim = 2
        self.qkv_proj = nn.Linear(2, 6, bias=False)
        self.q_norm = nn.RMSNorm(2)
        self.k_norm = nn.RMSNorm(2)
        self.out_proj = nn.Linear(2, 2, bias=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttn()


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.blocks = nn.ModuleList([Block()])


class FakePatcher:
    def __init__(self):
        self.model = Root()
        self.object_patches = {}
        self.wrappers = []
        self.patches = {}

    def clone(self):
        return copy.deepcopy(self)

    def get_model_object(self, key):
        if key in self.object_patches:
            return self.object_patches[key]
        value = self.model
        for part in key.split("."):
            value = getattr(value, part)
        return value

    def model_state_dict(self):
        return self.model.state_dict()

    def add_object_patch(self, key, value):
        self.object_patches[key] = value

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))

    def add_patches(self, patches, strength):
        self.patches.update(patches)
        return list(patches)


class Branch:
    w = {"to_out_linear.weight": torch.eye(2)}
    enable_text_state = False

    def readout(self, *args, **kwargs):
        raise AssertionError("not used")


def state():
    return VDNState("tiny", {"enable_softmax_gate": False}, [Branch()], 1, 2)


def test_clone_and_apply_does_not_mutate_source_modelpatcher():
    source = FakePatcher()
    result = clone_and_apply(source, state())
    assert source.object_patches == {}
    assert source.wrappers == []
    assert "diffusion_model.blocks.0.attn.forward" in result.object_patches
    assert "diffusion_model._vdn_h3_private_state" in result.object_patches


def test_existing_attention_patch_is_rejected_instead_of_overwritten():
    model = FakePatcher()
    model.object_patches["diffusion_model.blocks.0.attn.forward"] = lambda x: x
    with pytest.raises(RuntimeError, match="existing attention object patch"):
        apply_vdn(model, state())


def test_compact_coordinate_deltas_become_reversible_offset_patches():
    model = FakePatcher()
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    deltas = [
        DeltaPatch(key, torch.ones(2, 2), (0, 0)),
        DeltaPatch(key, torch.ones(2, 2), (2, 0)),
        DeltaPatch(key, torch.ones(2, 2), (4, 0)),
    ]
    assert apply_delta_patches(model, deltas) == 3
    assert {(entry[1]) for entry in model.patches} == {
        (0, 0, 2),
        (0, 2, 2),
        (0, 4, 2),
    }


def test_missing_or_bad_shape_adapter_target_fails_before_application():
    model = FakePatcher()
    with pytest.raises(KeyError, match="does not exist"):
        apply_delta_patches(model, [DeltaPatch("nope.weight", torch.ones(1))])
    assert model.patches == {}
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    with pytest.raises(ValueError, match="exceeds"):
        apply_delta_patches(model, [DeltaPatch(key, torch.ones(3, 2), (5, 0))])
    assert model.patches == {}
