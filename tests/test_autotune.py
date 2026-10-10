import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vdn_h3 import autotune, window
from vdn_h3.calibration import CALIBRATION_VERSION, CalibrationStore


def test_autotune_rejects_missing_anchor_columns_with_distinct_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'test')
    monkeypatch.setattr(autotune, '_signature', lambda *args: 'test-geometry')
    monkeypatch.setattr(autotune, '_timed', lambda fn, device, runs: (fn(), 1.0))
    monkeypatch.setattr(window, 'flex_available', lambda cache: False)
    monkeypatch.setattr(window, 'decomposed_available', lambda cache: False)
    monkeypatch.setattr(window, 'flash2_available', lambda cache: True)

    def missing_columns(q, k, v, **kwargs):
        kwargs.pop('cache')
        kwargs['anchor_frames'] = 'none'
        return window.window_softmax_grouped(q, k, v, **kwargs)

    monkeypatch.setattr(window, 'window_softmax_flash2', missing_columns)
    generator = torch.Generator().manual_seed(27)
    q, k = [F.rms_norm(torch.randn(13, 2, 128, generator=generator), (128,)) for _ in range(2)]
    v = torch.randn(13, 2, 128, generator=generator)
    layout = SimpleNamespace(video_start=2, video_end=13, num_frames=11,
                             tokens_per_frame=1, bounds=window.window_bounds(11, 1),
                             anchor_frames='columns')
    cache = window.WindowAttentionCache()
    cache.calibration = CalibrationStore(tmp_path / 'calibration.json')
    autotune.runtime_autotune_attention(q, q, q, layout, 128**-0.5, cache)
    assert cache.calibration._entries['test-geometry']['results']['flash2']['allclose']
    cache.calibration = CalibrationStore(tmp_path / 'distinct.json')
    assert autotune.runtime_autotune_attention(q, k, v, layout, 128**-0.5, cache) == 'grouped'
    results = cache.calibration._entries['test-geometry']['results']
    assert results['grouped']['allclose']
    assert not results['flash2']['allclose']


def test_runtime_calibration_receives_real_qkv_and_three_runs(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(window, 'flex_available', lambda cache: True)
    monkeypatch.setattr(window, 'flash2_available', lambda cache: False)
    monkeypatch.setattr(window, 'decomposed_available', lambda cache: False)
    q, k, v = [torch.randn(13, 2, 4) for _ in range(3)]
    calls = []

    def tune(query, key, value, layout, scale, cache, *, runs):
        assert query is q and key is k and value is v
        assert scale == 0.25
        assert runs == 3
        calls.append(layout)
        return 'grouped'

    monkeypatch.setattr(autotune, 'runtime_autotune_attention', tune)
    got = window._autotune_if_needed(q, 11, window.window_bounds(11, 1), 'columns',
                                    window.WindowAttentionCache(), 2, 13, 1, k, v, 0.25)
    assert got == 'grouped' and len(calls) == 1


def test_calibration_does_not_reuse_q_only_parity_results(tmp_path):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({'version': CALIBRATION_VERSION - 1,
                               'entries': {'geometry': {'winner': 'flash2'}}}))
    assert CalibrationStore(path).lookup('geometry') is None
