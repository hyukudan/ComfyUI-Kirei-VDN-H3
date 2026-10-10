from types import SimpleNamespace

import pytest
import torch

from vdn_h3 import branch, hybrid
from vdn_h3.diagnostics import DiagnosticsRecorder
from vdn_h3.layout import VDNLayout


def _inputs():
    generator = torch.Generator().manual_seed(541)
    features = torch.randn(3, 2, 4, 4, generator=generator)
    a = features @ features.transpose(-1, -2)
    return torch.rand(3, 2, 4, generator=generator), a, torch.randn_like(a)


@pytest.fixture
def deferred_cpu(monkeypatch):
    # Exercise the scope/error mechanism without requiring CUDA in CPU CI.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_native_cpu_and_outside_scope_keep_immediate_checks(deferred_cpu):
    alpha, a, b = _inputs()
    a[0, 1] = -2 * torch.eye(4)
    with torch.no_grad(), pytest.raises(torch.linalg.LinAlgError):
        branch.VdnDelta().factor_apply(alpha, a, b)
    assert branch._CHOLESKY_CHECKS.get() is None


def test_actual_cpu_scope_uses_native_cholesky():
    values = _inputs()
    with torch.no_grad(), branch.cholesky_runtime_scope(DiagnosticsRecorder()) as checks:
        branch.VdnDelta().factor_apply(*values)
        assert checks.factor_calls == 0 and not checks.statuses
        values[1][0, 1] = -2 * torch.eye(4)
        with pytest.raises(torch.linalg.LinAlgError):
            branch.VdnDelta().factor_apply(*values)
        assert checks.factor_calls == 0 and not checks.statuses


def test_deferred_factors_match_native_bit_for_bit(deferred_cpu):
    values = _inputs()
    diagnostics = DiagnosticsRecorder(True)
    with torch.no_grad():
        expected = branch.VdnDelta().factor_apply(*values)
        with branch.cholesky_runtime_scope(diagnostics) as checks:
            actual = branch.VdnDelta().factor_apply(*values)
            assert checks.factor_calls == 1 and len(checks.statuses) == 1
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    assert not checks.statuses and branch._CHOLESKY_CHECKS.get() is None
    assert diagnostics.snapshot()["stages"]["linear.cholesky_check"]["count"] == 1


@pytest.mark.parametrize("bad_diagonal", [-1.0, -2.0])
def test_bad_middle_factor_cannot_escape_forward(deferred_cpu, bad_diagonal):
    alpha, a, b = _inputs()
    bad = a.clone()
    bad[1, 1] = bad_diagonal * torch.eye(4)
    completed = 0
    with torch.no_grad(), pytest.raises(torch.linalg.LinAlgError, match="factor 1, batch 3"):
        with branch.cholesky_runtime_scope(DiagnosticsRecorder()) as checks:
            for matrix in (a, bad, a):
                branch.VdnDelta().factor_apply(alpha, matrix, b)
                completed += 1
    assert completed == 3 and checks.factor_calls == 3
    assert not checks.statuses and branch._CHOLESKY_CHECKS.get() is None


def test_nested_failure_restores_outer_scope_and_cancel_clears(deferred_cpu):
    values = _inputs()
    with torch.no_grad(), branch.cholesky_runtime_scope(DiagnosticsRecorder()) as outer:
        branch.VdnDelta().factor_apply(*values)
        with pytest.raises(RuntimeError, match="cancelled"):
            with branch.cholesky_runtime_scope(DiagnosticsRecorder()) as inner:
                branch.VdnDelta().factor_apply(*values)
                raise RuntimeError("cancelled")
        assert branch._CHOLESKY_CHECKS.get() is outer
        assert len(outer.statuses) == 1 and not inner.statuses
    assert not outer.statuses and branch._CHOLESKY_CHECKS.get() is None


@pytest.mark.parametrize("mode", ["autograd", "compile", "capture"])
def test_unsafe_modes_retain_native_checks(deferred_cpu, monkeypatch, mode):
    if mode == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif mode == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    alpha, a, b = _inputs()
    a[0, 1] = -2 * torch.eye(4)
    with torch.set_grad_enabled(mode == "autograd"):
        with branch.cholesky_runtime_scope(DiagnosticsRecorder()) as checks:
            with pytest.raises(torch.linalg.LinAlgError):
                branch.VdnDelta().factor_apply(alpha, a, b)
            assert checks.factor_calls == 0 and not checks.statuses


def test_wrapper_validates_before_return_and_records_counts(deferred_cpu, monkeypatch):
    layout = VDNLayout(seq_len=1, video_start=0, video_end=1, num_frames=1,
                       tokens_per_frame=1, frame_size=(1, 1), text_start=0, text_len=0)
    monkeypatch.setattr(hybrid, "layout_from_payload", lambda *args: layout)
    state = SimpleNamespace(config={}, forwards=0, curve_adapter=None,
                            diagnostics=DiagnosticsRecorder(True),
                            cholesky_deferred_factors=0, cholesky_validated_forwards=0)
    values = _inputs()

    def executor(*args):
        return branch.VdnDelta().factor_apply(*values)

    wrapped = hybrid.make_layout_wrapper(state)
    with torch.no_grad():
        wrapped(executor, torch.zeros(1), None, None)
    assert state.cholesky_deferred_factors == 1 and state.cholesky_validated_forwards == 1
    values[1][0, 1] = -2 * torch.eye(4)
    with torch.no_grad(), pytest.raises(torch.linalg.LinAlgError):
        wrapped(executor, torch.zeros(1), None, None)
    assert state.cholesky_deferred_factors == 1 and state.cholesky_validated_forwards == 1
    assert branch._CHOLESKY_CHECKS.get() is None


def test_nested_spd_error_does_not_invalidate_outer_scope(deferred_cpu):
    values = _inputs()
    invalid = tuple(value.clone() for value in values)
    invalid[1][0, 1] = -2 * torch.eye(4)
    with torch.no_grad(), branch.cholesky_runtime_scope(DiagnosticsRecorder()) as outer:
        branch.VdnDelta().factor_apply(*values)
        with pytest.raises(torch.linalg.LinAlgError):
            with branch.cholesky_runtime_scope(DiagnosticsRecorder()) as inner:
                branch.VdnDelta().factor_apply(*invalid)
        assert branch._CHOLESKY_CHECKS.get() is outer
        assert len(outer.statuses) == 1 and not inner.statuses
    assert not outer.statuses and branch._CHOLESKY_CHECKS.get() is None
