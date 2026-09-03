from types import SimpleNamespace

import pytest

from vdn_h3.layout import (
    VDNLayout,
    current_layout,
    from_comfy_packed_layout,
    publish_layout,
)


def packed(*, text=3, frames=2, height=4, width=4, audio_rows=2):
    per_frame = ((height + 1) // 2) * ((width + 1) // 2)
    video_start = text + audio_rows
    video_end = video_start + frames * per_frame
    return SimpleNamespace(
        signature=(text, frames, height, width, audio_rows // 2),
        segments=((0, text, "text"), (text, video_start, "audio"), (video_start, video_end, "video")),
        seq_len=video_end,
    )


def test_layout_is_strict_and_computes_full_cover():
    layout = from_comfy_packed_layout(packed(), radius=1, chunk=5, anchor_frames="both")
    assert layout.video_start == 5
    assert layout.video_end == 13
    assert layout.frame_size == (2, 2)
    assert layout.tokens_per_frame == 4
    assert layout.full_cover is True
    assert len(layout.bounds) == 2


def test_layout_supports_odd_prepadding_dimensions():
    layout = from_comfy_packed_layout(packed(frames=3, height=3, width=5), radius=0, chunk=0)
    assert layout.frame_size == (2, 3)
    assert layout.tokens_per_frame == 6
    assert layout.full_cover is False


def test_layout_rejects_inconsistent_video_geometry():
    value = packed()
    value.segments = value.segments[:-1] + ((5, 12, "video"),)
    value.seq_len = 12
    with pytest.raises(ValueError, match="not divisible"):
        from_comfy_packed_layout(value)


def test_contextvar_restores_nested_and_exception_state():
    outer = from_comfy_packed_layout(packed(frames=2))
    inner = from_comfy_packed_layout(packed(frames=3))
    assert current_layout(required=False) is None
    with publish_layout(outer):
        assert current_layout() is outer
        with pytest.raises(RuntimeError, match="boom"):
            with publish_layout(inner):
                assert current_layout() is inner
                raise RuntimeError("boom")
        assert current_layout() is outer
    assert current_layout(required=False) is None


def test_missing_context_has_clear_error():
    with pytest.raises(RuntimeError, match="without a packed layout"):
        current_layout()
