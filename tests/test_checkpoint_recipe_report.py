from types import SimpleNamespace

from vdn_h3.benchmark import _adapter_snapshot, _checkpoint_recipe


def test_checkpoint_recipe_exposes_declared_turbo_steps():
    state = SimpleNamespace(
        config={
            "turbo_num_steps": 8,
            "chunk": 5,
            "radius": 1,
            "anchor_frames": "both",
            "linear_head_dim": 128,
            "delta_rule": "vdn_solve",
            "bridge": "alpha",
            "unrelated": {"not": "serializable recipe metadata"},
        }
    )
    assert _checkpoint_recipe(state) == {
        "turbo_num_steps": 8,
        "chunk": 5,
        "radius": 1,
        "anchor_frames": "both",
        "linear_head_dim": 128,
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
    }


def test_adapter_snapshot_exposes_exact_active_recipe():
    state = SimpleNamespace(
        adapters={
            "active": ["default", "turbo"],
            "strengths": {"default": 1.0, "turbo": 1.0},
            "lora_mode": "bypass",
            "reports": ["default@1:bypass=3", "turbo@1:bypass=3"],
        }
    )
    assert _adapter_snapshot(state) == {
        "active": ["default", "turbo"],
        "strengths": {"default": 1.0, "turbo": 1.0},
        "lora_mode": "bypass",
        "reports": ["default@1:bypass=3", "turbo@1:bypass=3"],
    }
