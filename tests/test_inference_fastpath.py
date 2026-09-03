"""Regression tests for the allocation-aware VDN inference path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vdn_h3_private import branch
from vdn_h3_private.hybrid import ManagedBranchWeights, VDNState, make_vdn_forward
from vdn_h3_private.layout import VDNLayout, publish_layout
from vdn_h3_private.window import window_bounds


def test_frame_statistics_reuses_a_noncontiguous_key_without_changing_math():
    torch.manual_seed(101)
    base_key = torch.randn(4, 7, 3, 5)
    base_value = torch.randn_like(base_key)
    key = base_key.permute(0, 2, 1, 3)
    value = base_value.permute(0, 2, 1, 3)
    assert not key.is_contiguous()
    beta = torch.rand(4, 3, 7)

    matrix_a, matrix_b = branch.frame_statistics(key, value, beta)
    expected_a = torch.einsum("fhsk,fhsl,fhs->fhkl", key, key, beta)
    expected_b = torch.einsum("fhsv,fhsk,fhs->fhvk", value, key, beta)

    torch.testing.assert_close(matrix_a, expected_a, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(matrix_b, expected_b, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(matrix_a, matrix_a.transpose(-1, -2))


def test_preallocated_inference_scan_matches_the_differentiable_scan():
    torch.manual_seed(102)
    frames, heads, dim = 9, 3, 5
    features = torch.randn(frames, heads, 7, dim)
    matrix_a = features.transpose(-1, -2) @ features
    matrix_b = torch.randn(frames, heads, dim, dim)
    alpha = torch.rand(frames, heads, dim) * 0.7 + 0.2
    text_state = torch.randn(heads, dim, dim)
    backend = branch.VdnDelta()

    expected_prefix, expected_suffix = branch.run_scans(
        backend, alpha, matrix_a, matrix_b, text_state
    )
    with torch.no_grad():
        prefix, suffix = branch.run_scans_inference(
            backend, alpha, matrix_a, matrix_b, text_state
        )

    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(suffix, expected_suffix, atol=0, rtol=0)


def test_inference_scan_refuses_to_silently_break_autograd():
    frames, heads, dim = 3, 2, 4
    matrix_a = torch.eye(dim).expand(frames, heads, dim, dim).clone()
    matrix_b = torch.randn_like(matrix_a)
    alpha = torch.full((frames, heads, dim), 0.8)
    try:
        branch.run_scans_inference(branch.VdnDelta(), alpha, matrix_a, matrix_b)
    except RuntimeError as exc:
        assert "gradients" in str(exc)
    else:
        raise AssertionError("inference scan accepted grad-enabled execution")


def test_retention_weights_keep_checkpoint_precision_in_both_storage_modes():
    source = {
        "alpha.A_log": torch.ones(2, dtype=torch.float32),
        "alpha.dt_bias": torch.ones(8, dtype=torch.float32),
        "alpha.down.weight": torch.ones(4, 6, dtype=torch.float32),
        "alpha.up.weight": torch.ones(8, 4, dtype=torch.float32),
        "norm.weight": torch.ones(4, dtype=torch.float32),
    }
    for mode in ("stream", "resident"):
        store = ManagedBranchWeights([source], mode=mode)
        materialized = store.weights_on(0, "cpu", torch.bfloat16)
        for key in (
            "alpha.A_log",
            "alpha.dt_bias",
            "alpha.down.weight",
            "alpha.up.weight",
        ):
            assert materialized[key].dtype == torch.float32
        assert materialized["norm.weight"].dtype == torch.bfloat16
        store.close()


class _RMS(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _RecordingLinear(nn.Linear):
    last_output: torch.Tensor | None = None

    def forward(self, x):
        result = super().forward(x)
        self.last_output = result
        return result


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = _RecordingLinear(4, 12, bias=False)
        self.q_norm = _RMS(2)
        self.k_norm = _RMS(2)
        self.out_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x, rope_freqs=None, transformer_options=None):
        del rope_freqs, transformer_options
        q, k, v = self.qkv_proj(x).split(4, dim=-1)
        q = self.q_norm(q.view(-1, 2, 2))
        k = self.k_norm(k.view(-1, 2, 2))
        v = v.view(-1, 2, 2)
        attended = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=2**-0.5,
        ).squeeze(0).transpose(0, 1)
        return self.out_proj(attended.reshape(x.shape[0], 4))


class _RecordingBranch:
    enable_text_state = False

    def __init__(self):
        self.w = {"to_out_linear.weight": torch.eye(4)}
        self.raw = None

    def readout(self, weights, x, q_raw, k_raw, v_raw, *args, **kwargs):
        del weights, args, kwargs
        self.raw = (q_raw, k_raw, v_raw)
        return torch.zeros(x.shape[0], 4, device=x.device, dtype=x.dtype)


def _layout(frames: int) -> VDNLayout:
    return VDNLayout(
        seq_len=1 + frames,
        video_start=1,
        video_end=1 + frames,
        num_frames=frames,
        tokens_per_frame=1,
        frame_size=(1, 1),
        text_start=0,
        text_len=1,
        bounds=tuple(window_bounds(frames, 0, 0)),
        full_cover=False,
        anchor_frames="none",
    )


def test_linear_branch_reads_zero_copy_views_of_the_fused_qkv_projection():
    torch.manual_seed(103)
    attention = _TinyAttention()
    recording_branch = _RecordingBranch()
    state = VDNState(
        "tiny",
        {"enable_softmax_gate": False, "linear_enabled": True},
        [recording_branch],
        2,
        2,
    )
    forward = make_vdn_forward(attention, state, 0)
    x = torch.randn(5, 4)

    with torch.no_grad(), publish_layout(_layout(4)):
        result = forward(x)

    assert result.shape == (5, 4)
    projected = attention.qkv_proj.last_output
    assert projected is not None and recording_branch.raw is not None
    storage = projected.untyped_storage().data_ptr()
    for raw in recording_branch.raw:
        assert raw.untyped_storage().data_ptr() == storage
