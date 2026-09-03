"""VideoDeltaNet integration for native ComfyUI MiniMax-H3."""

__version__ = "0.3.0"  # keep in sync with pyproject.toml (tests/test_version.py)

from .layout import VDNLayout, current_layout, from_comfy_packed_layout, publish_layout

__all__ = [
    "__version__",
    "VDNLayout",
    "current_layout",
    "from_comfy_packed_layout",
    "publish_layout",
]
