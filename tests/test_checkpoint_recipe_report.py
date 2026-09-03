from types import SimpleNamespace

from vdn_h3.benchmark import _checkpoint_recipe


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
