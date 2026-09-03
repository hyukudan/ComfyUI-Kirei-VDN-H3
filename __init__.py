"""ComfyUI entry point for Kirei VDN-H3."""

# ComfyUI imports custom-node roots as packages, while pytest can collect this file
# as a top-level module because the repository directory contains hyphens.
try:
    from .vdn_h3.nodes import (
        NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as CORE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .vdn_h3.report_node import (
        NODE_CLASS_MAPPINGS as REPORT_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as REPORT_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    from vdn_h3.nodes import (
        NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as CORE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from vdn_h3.report_node import (
        NODE_CLASS_MAPPINGS as REPORT_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as REPORT_NODE_DISPLAY_NAME_MAPPINGS,
    )

NODE_CLASS_MAPPINGS = {**CORE_NODE_CLASS_MAPPINGS, **REPORT_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **CORE_NODE_DISPLAY_NAME_MAPPINGS,
    **REPORT_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
