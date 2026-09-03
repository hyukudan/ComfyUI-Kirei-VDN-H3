import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vdn_h3.adaln import TIME_EMBEDDER_PATCH, install_adaln_fp32


class Embedder(nn.Module):
    """Stands in for MiniMax-H3's fp32 TimeEmbedder."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 64)

    def forward(self, t):
        return self.proj(t.float()) * 7.0


class Adaln(nn.Module):
    """ComfyUI's AdalnProj.forward, with a BF16 projection like a loaded H3."""

    def __init__(self, hidden, expand, modalities):
        super().__init__()
        self.hidden, self.expand, self.modalities = hidden, expand, modalities
        self.apply_silu = True
        # 64 inputs: wide enough not to be mistaken for a pruned/curve AdaLN basis.
        self.linear = nn.Linear(64, expand * hidden * modalities).to(torch.bfloat16)

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _model(blocks=2, curve=False):
    dm = nn.Module()
    dm.time_embedder = Embedder()
    dm.blocks = nn.ModuleList()
    for _ in range(blocks):
        block = nn.Module()
        block.adaln_proj = Adaln(2, 6, 3)
        dm.blocks.append(block)
    dm.final_layer = nn.Module()
    dm.final_layer.adaln_proj = Adaln(2, 2, 1)
    if curve:
        dm.use_adaln_curves = True
    return dm


class Patcher:
    def __init__(self, dm):
        self.dm = dm
        self.object_patches = {}

    def get_model_object(self, key):
        parts = key.split(".")
        assert parts[0] == "diffusion_model"
        value = self.dm
        for part in parts[1:]:
            value = getattr(value, part)
        return value

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


class State:
    def __init__(self, enabled=True):
        self.adaln_fp32 = enabled
        self.adaln_source = None


def _expected(module, t32):
    # OpenVDN's diffusers patch: SiLU in fp32, cast to the projection dtype, project.
    activated = F.silu(t32).to(torch.bfloat16)
    x = module.linear(activated)
    x = x.view(x.shape[0] * module.modalities, module.expand * module.hidden)
    return x.chunk(module.expand, dim=-1)


def test_adaln_reads_the_fp32_embedding_like_the_upstream_patch():
    torch.manual_seed(3)
    dm = _model()
    patcher, state = Patcher(dm), State()
    assert install_adaln_fp32(patcher, state, dm) == 3
    assert state.adaln_fp32

    t = torch.randn(5, 4)
    t32 = patcher.object_patches[TIME_EMBEDDER_PATCH](t)
    assert t32.dtype == torch.float32
    assert state.adaln_source is t32 or torch.equal(state.adaln_source, t32)
    t_emb = t32.to(torch.bfloat16)  # what the model hands every block

    for path in ("blocks.0.adaln_proj", "blocks.1.adaln_proj", "final_layer.adaln_proj"):
        module = patcher.get_model_object(f"diffusion_model.{path}")
        got = patcher.object_patches[f"diffusion_model.{path}.forward"](t_emb)
        for actual, expected in zip(got, _expected(module, t32)):
            torch.testing.assert_close(actual, expected)
        native = module(t_emb)
        assert any(not torch.equal(a, b) for a, b in zip(got, native)), (
            "the fp32 activation should differ from ComfyUI's BF16 SiLU"
        )


def test_adaln_without_a_stashed_embedding_still_activates_in_fp32():
    torch.manual_seed(4)
    dm = _model(blocks=1)
    patcher, state = Patcher(dm), State()
    install_adaln_fp32(patcher, state, dm)
    module = dm.blocks[0].adaln_proj
    t_emb = (torch.randn(3, 64) * 7.0).to(torch.bfloat16)
    got = patcher.object_patches["diffusion_model.blocks.0.adaln_proj.forward"](t_emb)
    for actual, expected in zip(got, _expected(module, t_emb.float())):
        torch.testing.assert_close(actual, expected)


def test_adaln_patch_is_skipped_for_curve_bases_and_when_disabled():
    dm = _model(curve=True)
    patcher, state = Patcher(dm), State()
    assert install_adaln_fp32(patcher, state, dm) == 0
    assert not state.adaln_fp32 and not patcher.object_patches

    dm = _model()
    patcher, state = Patcher(dm), State(enabled=False)
    assert install_adaln_fp32(patcher, state, dm) == 0
    assert not state.adaln_fp32 and not patcher.object_patches

    dm = _model()
    patcher, state = Patcher(dm), State()
    patcher.object_patches["diffusion_model.blocks.0.adaln_proj.forward"] = object()
    with pytest.raises(RuntimeError, match="collides"):
        install_adaln_fp32(patcher, state, dm)
