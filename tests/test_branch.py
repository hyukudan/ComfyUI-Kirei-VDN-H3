"""Independent numerical tests for the VDN bidirectional linear branch."""

from __future__ import annotations

import torch

from vdn_h3_private import branch
from vdn_h3_private.window import window_bounds


def test_vdn_delta_matches_closed_form_for_psd_statistics():
    torch.manual_seed(1)
    frames, heads, dim, tokens = 4, 2, 5, 7
    key = torch.randn(frames, heads, tokens, dim)
    beta = torch.rand(frames, heads, tokens)
    matrix_a = (key * beta.unsqueeze(-1)).transpose(-1, -2) @ key
    matrix_b = torch.randn(frames, heads, dim, dim)
    alpha = torch.rand(frames, heads, dim)
    transition, injection = branch.VdnDelta().factor_apply(alpha, matrix_a, matrix_b)
    identity = torch.eye(dim).expand_as(matrix_a)
    inverse = torch.linalg.inv(identity + matrix_a)
    torch.testing.assert_close(transition, alpha.unsqueeze(-1) * inverse, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(injection, matrix_b @ inverse, atol=3e-5, rtol=3e-5)


def test_frame_statistics_matches_direct_equations():
    torch.manual_seed(2)
    key = torch.randn(3, 2, 6, 4)
    value = torch.randn_like(key)
    beta = torch.rand(3, 2, 6)
    matrix_a, matrix_b = branch.frame_statistics(key, value, beta)
    expected_a = torch.einsum("fhsk,fhsl,fhs->fhkl", key, key, beta)
    expected_b = torch.einsum("fhsv,fhsk,fhs->fhvk", value, key, beta)
    torch.testing.assert_close(matrix_a, expected_a)
    torch.testing.assert_close(matrix_b, expected_b)
    torch.testing.assert_close(matrix_a, matrix_a.transpose(-1, -2))


def test_bidirectional_scans_match_naive_recurrence():
    torch.manual_seed(3)
    frames, heads, dim = 5, 2, 4
    alpha = torch.rand(frames, heads, dim) * 0.5 + 0.25
    features = torch.randn(frames, heads, 6, dim)
    matrix_a = features.transpose(-1, -2) @ features
    matrix_b = torch.randn(frames, heads, dim, dim)
    text = torch.randn(heads, dim, dim)
    backend = branch.VdnDelta()
    transition, injection = backend.factor_apply(alpha, matrix_a, matrix_b)

    expected_forward = []
    state = text
    for frame in range(frames):
        state = state @ transition[frame] + injection[frame]
        expected_forward.append(state)
    expected_reverse = [None] * frames
    state = text
    for frame in range(frames - 1, -1, -1):
        state = state @ transition[frame] + injection[frame]
        expected_reverse[frame] = state
    got_forward, got_reverse = branch.run_scans(
        backend, alpha, matrix_a, matrix_b, text_state=text
    )
    torch.testing.assert_close(got_forward, torch.stack(expected_forward), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got_reverse, torch.stack(expected_reverse), atol=2e-5, rtol=2e-5)


def _naive_gather(prefix, suffix, alpha, bounds, text=None):
    frames = prefix.shape[0]
    log_prefix = torch.cat((torch.zeros_like(alpha[:1]), alpha.clamp_min(1e-12).log().cumsum(0)))
    output = []
    zero = torch.zeros_like(prefix[0])
    for frame, (raw_lo, raw_hi) in enumerate(bounds):
        lo, hi = max(raw_lo, 0), min(raw_hi, frames - 1)
        if lo:
            before = prefix[lo - 1]
        else:
            before = zero if text is None else text
        if hi + 1 < frames:
            after = suffix[hi + 1]
        else:
            after = zero if text is None else text
        before = before * torch.exp(log_prefix[frame + 1] - log_prefix[lo]).unsqueeze(1)
        after = after * torch.exp(log_prefix[hi + 1] - log_prefix[frame]).unsqueeze(1)
        output.append(before + after)
    return torch.stack(output)


def test_gather_matches_naive_with_and_without_text():
    torch.manual_seed(4)
    frames, heads, dim = 13, 2, 3
    prefix = torch.randn(frames, heads, dim, dim)
    suffix = torch.randn_like(prefix)
    alpha = torch.rand(frames, heads, dim) * 0.8 + 0.1
    bounds = window_bounds(frames, 1, 4)
    text = torch.randn(heads, dim, dim)
    for initial in (None, text):
        got = branch.gather_linear_state(
            prefix, suffix, alpha, bounds, text_state=initial
        )
        want = _naive_gather(prefix, suffix, alpha, bounds, initial)
        torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)


def test_gather_cache_is_branch_owned_lru_and_keyed_by_gpu_index():
    cache = branch.GatherIndexCache(limit=8)
    frames = 48
    for radius in range(16):
        branch.gather_indices(
            window_bounds(frames, radius), frames, "cpu", cache=cache
        )
    assert len(cache) == 8
    assert branch._device_key(torch.device("cuda:0")) != branch._device_key(torch.device("cuda:1"))
    cache.release()
    assert len(cache) == 0
    assert not hasattr(branch, "_GATHER_INDEX_CACHE")


def test_conv_features_and_temporal_shift_shapes():
    torch.manual_seed(5)
    frames, height, width, heads, dim = 3, 2, 3, 2, 2
    channels = heads * dim
    tokens = torch.randn(frames * height * width, heads, dim)
    spatial = torch.randn(channels, 1, 3, 3)
    temporal = torch.randn(channels, 1, 3)
    result = branch.conv_features(
        tokens, spatial, temporal, frames, (height, width), l2norm=True
    )
    assert result.shape == tokens.shape
    torch.testing.assert_close(
        torch.linalg.vector_norm(result, dim=-1),
        torch.ones(result.shape[:-1]),
        atol=2e-5,
        rtol=2e-5,
    )


def _small_weights(hidden, heads, dim, bottleneck=3):
    return {
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


def test_linear_branch_end_to_end_and_anchor_zeroing_is_differentiable():
    torch.manual_seed(6)
    frames, per_frame, heads, dim, hidden = 6, 4, 2, 3, 7
    rows = frames * per_frame
    weights = _small_weights(hidden, heads, dim)
    module = branch.LinearBranch(
        weights, heads, dim, short_conv=(), enable_text_state=True
    )
    xv = torch.randn(rows, hidden, requires_grad=True)
    q = torch.randn(rows, heads, dim, requires_grad=True)
    k = torch.randn(rows, heads, dim, requires_grad=True)
    v = torch.randn(rows, heads, dim, requires_grad=True)
    text_x = torch.randn(3, hidden)
    text_k = torch.randn(3, heads, dim)
    text_v = torch.randn(3, heads, dim)
    bounds = window_bounds(frames, 1, 2)
    result = module.readout(
        weights, xv, q, k, v, frames, per_frame, bounds,
        text_x=text_x, text_k_raw=text_k, text_v_raw=text_v,
        skip_ends=True,
    )
    assert result.shape == (rows, heads * dim)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[:per_frame]) == 0
    assert torch.count_nonzero(result[-per_frame:]) == 0
    result.square().mean().backward()
    assert xv.grad is not None and torch.isfinite(xv.grad).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert len(module._gather_cache) > 0
    module.release()
    assert len(module._gather_cache) == 0
    assert not module._backends


def test_source_has_no_tensor_scalar_sync_item_call():
    import inspect

    assert ".item(" not in inspect.getsource(branch)
