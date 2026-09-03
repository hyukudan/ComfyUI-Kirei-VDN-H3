"""ComfyUI entry point for the optimized Kirei VDN-H3 integration."""

# ComfyUI imports custom-node roots as packages, while pytest can collect this file
# as a top-level module because the directory name contains hyphens.
try:
    from .vdn_h3_private.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from vdn_h3_private.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
