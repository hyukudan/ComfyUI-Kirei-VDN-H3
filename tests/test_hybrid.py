from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from vdn_h3_private.hybrid import ManagedBranchWeights, VDNState, make_vdn_forward
from vdn_h3_private.layout import VDNLayout, publish_layout
from vdn_h3_private.window import window_bounds


class RMS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = RMS(2)
        self.k_norm = RMS(2)
        self.out_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x, rope_freqs=None, transformer_options=None):
        q, k, v = self.qkv_proj(x).split(4, dim=-1)
        q = self.q_norm(q.view(-1, 2, 2))
        k = self.k_norm(k.view(-1, 2, 2))
        v = v.view(-1, 2, 2)
        y = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=2**-0.5,
        ).squeeze(0).transpose(0, 1)
        return self.out_proj(y.reshape(x.shape[0], 4))


class Branch:
    enable_text_state = False

    def __init__(self, gate=True):
        self.w = {"to_out_linear.weight": torch.eye(4)}
        if gate:
            self.w.update(
                {
                    "softmax_gate.up.weight": torch.zeros(2, 4),
                    "softmax_gate.up.bias": torch.zeros(2),
                }
            )

    def readout(self, weights, x, *args, **kwargs):
        return torch.ones(x.shape[0], 4, device=x.device, dtype=x.dtype)


def layout(frames, *, full):
    bounds = tuple(window_bounds(frames, 8 if full else 0, 0))
    return VDNLayout(
        seq_len=1 + frames,
        video_start=1,
        video_end=1 + frames,
        num_frames=frames,
        tokens_per_frame=1,
        frame_size=(1, 1),
        text_start=0,
        text_len=1,
        bounds=bounds,
        full_cover=full,
        anchor_frames="none",
    )


def test_full_cover_gate_keeps_canonical_shape_and_loads_only_gate(monkeypatch):
    torch.manual_seed(4)
    attn = TinyAttention()
    attn.out_proj.weight.data.copy_(torch.eye(4))
    x = torch.randn(4, 4)
    teacher = attn(x)
    state = VDNState("tiny", {"enable_softmax_gate": True}, [Branch()], 2, 2)
    requested = []
    original = state.weights_on

    def recording_weights(index, device, dtype, keys=None):
        requested.append(keys)
        return original(index, device, dtype, keys)

    monkeypatch.setattr(state, "weights_on", recording_weights)
    forward = make_vdn_forward(attn, state, 0)
    with publish_layout(layout(3, full=True)):
        result = forward(x)
    assert result.shape == (4, 4)
    torch.testing.assert_close(result, teacher * 0.5)
    assert requested == [{"softmax_gate.up.weight", "softmax_gate.up.bias"}]


def test_full_cover_without_gate_matches_dense_teacher(monkeypatch):
    torch.manual_seed(8)
    attn = TinyAttention()
    x = torch.randn(3, 4)
    teacher = attn(x)
    state = VDNState("tiny", {"enable_softmax_gate": False}, [Branch(False)], 2, 2)
    monkeypatch.setattr(
        state,
        "weights_on",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full-cover without a gate must not transfer branch weights")
        ),
    )
    with publish_layout(layout(2, full=True)):
        result = make_vdn_forward(attn, state, 0)(x)
    torch.testing.assert_close(result, teacher)


def test_linear_delta_changes_only_target_video_rows():
    torch.manual_seed(12)
    attn = TinyAttention()
    attn.out_proj.weight.data.copy_(torch.eye(4))
    x = torch.randn(5, 4)
    branch = Branch(False)
    off = VDNState(
        "off", {"enable_softmax_gate": False, "linear_enabled": False}, [branch], 2, 2
    )
    on = VDNState(
        "on", {"enable_softmax_gate": False, "linear_enabled": True}, [branch], 2, 2
    )
    lay = layout(4, full=False)
    with publish_layout(lay):
        baseline = make_vdn_forward(attn, off, 0)(x)
        result = make_vdn_forward(attn, on, 0)(x)
    torch.testing.assert_close(result[:1], baseline[:1])
    torch.testing.assert_close(result[1:], baseline[1:] + 1)


def test_managed_store_is_per_instance_and_releasable():
    first = ManagedBranchWeights([{"w": torch.ones(2)}], mode="resident")
    second = ManagedBranchWeights([{"w": torch.zeros(2)}], mode="resident")
    first.weights_on(0, "cpu", torch.float32)["w"].add_(3)
    assert second.weights_on(0, "cpu", torch.float32)["w"].sum() == 0
    first.release()
    assert first.weights_on(0, "cpu", torch.float32)["w"].device.type == "cpu"
    first.close()
    assert first.closed


def test_managed_store_can_materialize_only_requested_weights():
    store = ManagedBranchWeights(
        [{"small": torch.ones(1), "large": torch.ones(1024)}], mode="stream"
    )
    selected = store.weights_on(0, "cpu", torch.float32, keys={"small"})
    assert set(selected) == {"small"}
