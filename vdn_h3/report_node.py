"""ComfyUI runtime-report node for Kirei VDN-H3."""

from __future__ import annotations

import json

from .benchmark import runtime_snapshot


class KireiVDNH3RuntimeReport:
    """Expose the resolved VDN runtime as JSON while passing the MODEL through.

    Connect any sampler/decoder output to the optional ``after`` input when the report
    must describe a completed render.  The value is ignored; the dependency exists
    only to make ComfyUI schedule this node after the connected producer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "after": (
                    "*",
                    {
                        "tooltip": (
                            "Optional dependency trigger. Connect the sampler LATENT or another "
                            "downstream output here to capture metrics after rendering."
                        )
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report_json")
    FUNCTION = "report"
    CATEGORY = "model_patches/video/advanced"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Return a JSON snapshot of the active VDN-H3 profile, backend, branch memory, "
        "CUDA memory and diagnostics. Connect a sampler output to 'after' for a "
        "post-render report. The MODEL is passed through unchanged."
    )

    def report(self, model, after=None):
        del after
        snapshot = runtime_snapshot(model)
        return model, json.dumps(snapshot, indent=2, sort_keys=True, default=str)


NODE_CLASS_MAPPINGS = {"KireiVDNH3RuntimeReport": KireiVDNH3RuntimeReport}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiVDNH3RuntimeReport": "Kirei VDN-H3 Runtime Report",
}


__all__ = [
    "KireiVDNH3RuntimeReport",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
