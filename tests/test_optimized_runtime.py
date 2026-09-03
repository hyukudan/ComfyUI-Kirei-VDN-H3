import torch
import torch.nn as nn
import torch.nn.functional as F

from vdn_h3 import branch, window
from vdn_h3.bypass import CompositeQKVBypassAdapter, FrugalLoRABypassAdapter
from vdn_h3.hybrid import VDNState, make_vdn_forward
from vdn_h3.layout import VDNLayout, publish_layout
from vdn_h3.weights import ManagedBranchWeights


def test_inference_scan_matches_reference():
    torch.manual_seed(1)
    frames, heads, dim = 5, 2, 4
    keys = torch.randn(frames, heads, 6, dim)
    beta = torch.rand(frames, heads, 6)
    matrix_a = (keys * beta[..., None]).transpose(-1, -2) @ keys
    matrix_b = torch.randn(frames, heads, dim, dim)
    alpha = torch.rand(frames, heads, dim) * 0.5 + 0.25
    backend = branch.VdnDelta()
    reference = branch.run_scans_reference(backend, alpha, matrix_a, matrix_b)
    with torch.no_grad():
        inference = branch.run_scans_inference(backend, alpha, matrix_a, matrix_b)
    for expected, actual in zip(reference, inference):
        torch.testing.assert_close(actual, expected)


def test_reference_branch_keeps_autograd():
    torch.manual_seed(2)
    frames, per_frame, heads, dim, hidden, bottleneck = 5, 3, 2, 3, 7, 3
    rows = frames * per_frame
    weights = {
        "beta_proj.weight": torch.randn(heads, hidden) * 0.1,
        "alpha.down.weight": torch.randn(bottleneck, hidden) * 0.1,
        "alpha.up.weight": torch.randn(heads * dim, bottleneck) * 0.1,
        "alpha.dt_bias": torch.randn(heads * dim) * 0.1,
        "alpha.A_log": torch.zeros(heads),
        "output_gate.down.weight": torch.randn(bottleneck, hidden) * 0.1,
        "output_gate.up.weight": torch.randn(heads * dim, bottleneck) * 0.1,
        "output_gate.up.bias": torch.randn(heads * dim) * 0.1,
        "norm.weight": torch.ones(dim),
    }
    module = branch.LinearBranch(weights, heads, dim, short_conv=(), enable_text_state=False)
    x = torch.randn(rows, hidden, requires_grad=True)
    q = torch.randn(rows, heads, dim, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = module.readout(weights, x, q, k, v, frames, per_frame, window.window_bounds(frames, 1, 2))
    out.square().mean().backward()
    assert x.grad is not None and q.grad is not None


def test_decomposed_plan_matches_window_oracle_for_all_anchor_modes():
    frames, per_frame, video_start, after = 11, 3, 4, 2
    video_end = video_start + frames * per_frame
    sequence = video_end + after
    for anchors in window.ANCHOR_FRAME_MODES:
        for radius, chunk in ((0, 0), (1, 3), (2, 4)):
            bounds = window.window_bounds(frames, radius, chunk)
            plan = window._DecomposedPlan(
                sequence, video_start, video_end, frames, per_frame,
                bounds, anchors, torch.device("cpu"),
            )
            allowed = {}
            for row in plan.dense_q.tolist():
                allowed[row] = set(range(sequence))
            if plan.has_windows:
                cu_q, cu_k = plan.cu_q.tolist(), plan.cu_k.tolist()
                win_q, gathered = plan.win_q.tolist(), plan.kv_gather.tolist()
                for group in range(len(cu_q) - 1):
                    keys = set(gathered[cu_k[group] : cu_k[group + 1]])
                    for row in win_q[cu_q[group] : cu_q[group + 1]]:
                        allowed[row] = keys
            for row in range(sequence):
                if row < video_start or row >= video_end:
                    expected = set(range(sequence))
                else:
                    frame = (row - video_start) // per_frame
                    if anchors in ("rows", "both") and frame in (0, frames - 1):
                        expected = set(range(sequence))
                    else:
                        lo, hi = bounds[frame]
                        selected = set(range(max(lo, 0), min(hi, frames - 1) + 1))
                        if anchors in ("columns", "both"):
                            selected.update((0, frames - 1))
                        expected = set(range(video_start)) | set(range(video_end, sequence))
                        for key_frame in selected:
                            expected.update(
                                range(
                                    video_start + key_frame * per_frame,
                                    video_start + (key_frame + 1) * per_frame,
                                )
                            )
                assert allowed[row] == expected


def test_fp32_branch_state_policy():
    store = ManagedBranchWeights(
        [{"alpha.A_log": torch.ones(2), "alpha.dt_bias": torch.ones(4), "other": torch.ones(3)}],
        mode="stream",
    )
    got = store.weights_on(0, "cpu", torch.bfloat16)
    assert got["alpha.A_log"].dtype == torch.float32
    assert got["alpha.dt_bias"].dtype == torch.float32
    assert got["other"].dtype == torch.bfloat16


def test_frugal_lora_matches_dense_delta():
    torch.manual_seed(30)
    x = torch.randn(7, 9)
    down = torch.randn(3, 9)
    up = torch.randn(5, 3)
    scale = 0.375
    adapter = FrugalLoRABypassAdapter(up, down, scale)
    got = adapter.delta(x)
    expected = F.linear(x, (up @ down) * scale)
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-6)


def test_composite_bypass_matches_independent_dense_qkv_deltas():
    torch.manual_seed(31)
    x = torch.randn(6, 8)
    base = torch.randn(6, 24)
    original = base.clone()
    specs = []
    expected = original.clone()
    for index, scale in enumerate((0.25, 0.5, 0.75)):
        down = torch.randn(2, 8)
        up = torch.randn(8, 2)
        adapter = FrugalLoRABypassAdapter(up, down, scale)
        specs.append((adapter, (index * 8, 8)))
        expected[:, index * 8 : (index + 1) * 8].add_(F.linear(x, (up @ down) * scale))
    got = CompositeQKVBypassAdapter(specs).apply(x, base)
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-6)


def test_composite_bypass_writes_only_qkv_slices():
    x = torch.randn(3, 4)
    base = torch.zeros(3, 12)
    q = FrugalLoRABypassAdapter(torch.ones(4, 1), torch.ones(1, 4), 0.5)
    v = FrugalLoRABypassAdapter(torch.ones(4, 1) * 2, torch.ones(1, 4), 0.25)
    adapter = CompositeQKVBypassAdapter([(q, (0, 4)), (v, (8, 4))])
    out = adapter.apply(x, base)
    assert torch.count_nonzero(out[:, 4:8]) == 0
    assert torch.count_nonzero(out[:, :4])
    assert torch.count_nonzero(out[:, 8:])


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
        q, k, v = self.qkv_proj(x).split(4, -1)
        q, k, v = self.q_norm(q.view(-1, 2, 2)), self.k_norm(k.view(-1, 2, 2)), v.view(-1, 2, 2)
        y = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], scale=2**-0.5
        )[0].transpose(0, 1)
        return self.out_proj(y.reshape(len(x), 4))


class TinyBranch:
    enable_text_state = False

    def __init__(self):
        self.w = {
            "to_out_linear.weight": torch.eye(4),
            "softmax_gate.up.weight": torch.zeros(2, 4),
            "softmax_gate.up.bias": torch.zeros(2),
        }

    def readout(self, weights, x, *args, **kwargs):
        return torch.ones(len(x), 4)


def _full_layout(frames):
    return VDNLayout(
        1 + frames, 1, 1 + frames, frames, 1, (1, 1), 0, 1,
        tuple(window.window_bounds(frames, 8)), True, "none",
    )


def test_full_cover_gate_keeps_teacher_shape_and_value():
    torch.manual_seed(4)
    attention = TinyAttention()
    attention.out_proj.weight.data.copy_(torch.eye(4))
    x = torch.randn(4, 4)
    teacher = attention(x)
    state = VDNState("tiny", {"enable_softmax_gate": True}, [TinyBranch()], 2, 2)
    with publish_layout(_full_layout(3)):
        got = make_vdn_forward(attention, state, 0)(x)
    torch.testing.assert_close(got, teacher * 0.5)
