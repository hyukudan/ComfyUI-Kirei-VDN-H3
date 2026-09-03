"""Strict MiniMax-H3 packed-layout adaptation and request-local publication.

ComfyUI does not pass the packed layout down to an attention module.  A
``ContextVar`` bridges that gap without putting mutable request state on the shared
model (which would be unsafe when previews or concurrent queues overlap).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Iterator, Sequence


_ANCHOR_MODES = frozenset({"none", "columns", "rows", "both"})


@dataclass(frozen=True, slots=True)
class VDNLayout:
    """Geometry needed by both branches of Video Delta Attention."""

    seq_len: int
    video_start: int
    video_end: int
    num_frames: int
    tokens_per_frame: int
    frame_size: tuple[int, int]
    text_start: int
    text_len: int
    bounds: tuple[tuple[int, int], ...] = ()
    full_cover: bool = False
    anchor_frames: str = "both"

    @property
    def frame_height(self) -> int:
        return self.frame_size[0]

    @property
    def frame_width(self) -> int:
        return self.frame_size[1]


CURRENT_LAYOUT: ContextVar[VDNLayout | None] = ContextVar(
    "vdn_h3_private_layout", default=None
)


def current_layout(*, required: bool = True) -> VDNLayout | None:
    """Return this execution context's layout, never another request's layout."""

    layout = CURRENT_LAYOUT.get()
    if required and layout is None:
        raise RuntimeError(
            "VDN-H3 attention ran without a packed layout. Apply the model through "
            "the VDN-H3 node and do not invoke a patched attention block directly."
        )
    return layout


@contextmanager
def publish_layout(layout: VDNLayout) -> Iterator[VDNLayout]:
    """Publish *layout* for one model forward and restore nested prior state."""

    if not isinstance(layout, VDNLayout):
        raise TypeError(f"expected VDNLayout, got {type(layout).__name__}")
    token: Token[VDNLayout | None] = CURRENT_LAYOUT.set(layout)
    try:
        yield layout
    finally:
        CURRENT_LAYOUT.reset(token)


def _one_segment(segments: Sequence[Any], kind: str) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) < 3:
            raise TypeError(f"invalid PackedLayout segment {segment!r}")
        start, stop, value = segment[:3]
        if value == kind:
            matches.append((int(start), int(stop)))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind!r} segment, got {len(matches)}")
    start, stop = matches[0]
    if start < 0 or stop <= start:
        raise ValueError(f"invalid {kind!r} segment [{start}, {stop})")
    return start, stop


def _window_geometry(
    num_frames: int, radius: int, chunk: int
) -> tuple[tuple[tuple[int, int], ...], bool]:
    try:
        from .window import full_coverage, window_bounds
    except ImportError:
        # This keeps layout.py independently testable while the optional optimized
        # window module is being installed.
        def window_bounds(n: int, r: int, c: int = 0):
            if c <= 0:
                return [(t - r, t + r) for t in range(n)]
            return [
                (((t // c) - r) * c, ((t // c) + r + 1) * c - 1)
                for t in range(n)
            ]

        def full_coverage(items, n: int):
            return all(lo <= 0 and hi >= n - 1 for lo, hi in items)

    bounds = tuple((int(lo), int(hi)) for lo, hi in window_bounds(num_frames, radius, chunk))
    if len(bounds) != num_frames:
        raise ValueError(
            f"window implementation returned {len(bounds)} bounds for {num_frames} frames"
        )
    return bounds, bool(full_coverage(bounds, num_frames))


def from_comfy_packed_layout(
    packed: Any,
    *,
    radius: int = 1,
    chunk: int = 5,
    anchor_frames: str = "both",
) -> VDNLayout:
    """Strictly adapt ComfyUI's ``PackedLayout`` without sequence guessing."""

    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError(f"window radius must be a non-negative integer, got {radius!r}")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 0:
        raise ValueError(f"window chunk must be a non-negative integer, got {chunk!r}")
    if anchor_frames not in _ANCHOR_MODES:
        raise ValueError(
            f"anchor_frames must be one of {sorted(_ANCHOR_MODES)}, got {anchor_frames!r}"
        )

    try:
        signature = tuple(map(int, packed.signature))
        segments = tuple(packed.segments)
        seq_len = int(packed.seq_len)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("not a compatible ComfyUI MiniMax-H3 PackedLayout") from exc
    if len(signature) != 5:
        raise ValueError(
            "PackedLayout.signature must be (text_len, latent_t, latent_h, "
            f"latent_w, audio_t), got {signature!r}"
        )
    text_len, latent_t, latent_h, latent_w, _audio_t = signature
    if min(seq_len, text_len, latent_t, latent_h, latent_w) <= 0:
        raise ValueError(f"invalid PackedLayout signature/length: {signature!r}, seq={seq_len}")

    text_start, text_end = _one_segment(segments, "text")
    video_start, video_end = _one_segment(segments, "video")
    if text_end - text_start != text_len:
        raise ValueError(
            "text segment length disagrees with PackedLayout.signature: "
            f"{text_end - text_start} != {text_len}"
        )
    if video_end > seq_len or text_end > seq_len:
        raise ValueError("a PackedLayout segment extends past the packed sequence")
    video_rows = video_end - video_start
    if video_rows % latent_t:
        raise ValueError(
            f"{video_rows} video rows are not divisible by {latent_t} latent frames"
        )
    tokens_per_frame = video_rows // latent_t
    # H3 pads the latent spatial axes, then applies a 2x2 patch.  Ceil division
    # handles a test/layout object supplied before padding as well as Comfy's normal
    # already-padded signature.
    frame_size = ((latent_h + 1) // 2, (latent_w + 1) // 2)
    if frame_size[0] * frame_size[1] != tokens_per_frame:
        raise ValueError(
            f"patched grid {frame_size[0]}x{frame_size[1]} does not match "
            f"{tokens_per_frame} video tokens per frame"
        )
    bounds, full_cover = _window_geometry(latent_t, radius, chunk)
    return VDNLayout(
        seq_len=seq_len,
        video_start=video_start,
        video_end=video_end,
        num_frames=latent_t,
        tokens_per_frame=tokens_per_frame,
        frame_size=frame_size,
        text_start=text_start,
        text_len=text_len,
        bounds=bounds,
        full_cover=full_cover,
        anchor_frames=anchor_frames,
    )


def layout_from_payload(
    payload: dict[str, Any] | None,
    x: Any,
    context: Any,
    config: dict[str, Any],
) -> VDNLayout:
    """Use or rebuild exactly the ``PackedLayout`` that native H3 consumes."""

    payload = payload or {}
    packed = payload.get("layout")
    try:
        video, audio = x[0], x[1]
        text_len = int(context.shape[1])
        latent_t = int(video.shape[2])
        latent_h = int(video.shape[3])
        latent_w = int(video.shape[4])
        audio_t = int(audio.shape[-1])
    except (IndexError, AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "VDN-H3 expected x=(video[B,C,T,H,W], audio[...,T]) and "
            "context[B,text,hidden]"
        ) from exc

    # Native H3 pads H/W to a multiple of its spatial patch size before creating
    # PackedLayout.  Avoid importing ComfyUI when a valid layout was supplied.
    padded_h, padded_w = (latent_h + 1) // 2 * 2, (latent_w + 1) // 2 * 2
    signature = (text_len, latent_t, padded_h, padded_w, audio_t)
    if packed is None or tuple(getattr(packed, "signature", ())) != signature:
        try:
            from comfy.ldm.minimax.model import PackedLayout
        except ImportError as exc:
            raise RuntimeError(
                "minimax_payload has no layout matching this input and ComfyUI's "
                "MiniMax-H3 PackedLayout is unavailable"
            ) from exc
        packed = PackedLayout(
            *signature,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )
    return from_comfy_packed_layout(
        packed,
        radius=int(config.get("radius", 1)),
        chunk=int(config.get("chunk", 5)),
        anchor_frames=str(config.get("anchor_frames", "both")),
    )


def with_window(layout: VDNLayout, *, radius: int, chunk: int) -> VDNLayout:
    """Return a layout with recomputed window metadata (useful for diagnostics)."""

    bounds, full_cover = _window_geometry(layout.num_frames, radius, chunk)
    return replace(layout, bounds=bounds, full_cover=full_cover)


__all__ = [
    "CURRENT_LAYOUT",
    "VDNLayout",
    "current_layout",
    "from_comfy_packed_layout",
    "layout_from_payload",
    "publish_layout",
    "with_window",
]
