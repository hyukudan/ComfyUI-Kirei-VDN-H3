import hashlib
import json

import pytest
import torch

from vdn_h3_private.spec import (
    inventory_checkpoint,
    load_vdn_checkpoint,
    resolve_vdn_checkpoint,
    transform_config,
    validate_model_spec,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _adapter_entry(module):
    return {
        "type": "lora",
        "version": 1,
        "config": {
            "rank": 1,
            "alpha": 1,
            "targets": [module],
            "exact_targets": True,
        },
    }


def _model_spec(*, adapter=None, num_layers=1):
    resolved = {
        "hidden_size": 4,
        "num_layers": num_layers,
        "num_attention_heads": 2,
        "attention_head_dim": 2,
    }
    config_hash = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "format_version": 2,
        "base": {
            "library": "diffusers",
            "class_name": "MiniMaxH3Transformer2DModel",
            "source": "example/base",
            "subfolder": "transformer",
            "revision": "fixed",
            "resolved_config": resolved,
            "config_hash": config_hash,
        },
        "transforms": [{
            "type": "hybrid_attention",
            "version": 2,
            "config": {
                "enable_softmax_gate": True,
                "anchor_frames": "both",
                "softmax_attention": {"radius": 1, "chunk": 5},
                "linear_attention": {
                    "delta_rule": "vdn_solve",
                    "bridge": "alpha",
                    "a_fp32": True,
                    "linear_head_dim": 2,
                    "short_conv": {"targets": ["k", "v"]},
                    "enable_text_state": True,
                },
            },
        }],
        "adapters": [] if adapter is None else [adapter],
    }


def _branch_state():
    prefix = "transformer_blocks.0.attn."
    shapes = {
        "to_out_linear.weight": (4, 4),
        "linear_attention.beta_proj.weight": (2, 4),
        "linear_attention.norm.weight": (2,),
        "linear_attention.alpha.A_log": (2,),
        "linear_attention.alpha.dt_bias": (4,),
        "linear_attention.alpha.down.weight": (2, 4),
        "linear_attention.alpha.up.weight": (4, 2),
        "linear_attention.output_gate.down.weight": (2, 4),
        "linear_attention.output_gate.up.weight": (4, 2),
        "linear_attention.output_gate.up.bias": (4,),
        "softmax_gate.up.weight": (2, 4),
        "softmax_gate.up.bias": (2,),
        "linear_attention.short_conv.k_sp.weight": (4, 1, 5, 5),
        "linear_attention.short_conv.k_tm.weight": (4, 1, 5),
        "linear_attention.short_conv.v_sp.weight": (4, 1, 5, 5),
        "linear_attention.short_conv.v_tm.weight": (4, 1, 5),
    }
    return {prefix + name: torch.zeros(shape) for name, shape in shapes.items()}


def _checkpoint(tmp_path, with_adapter=True):
    root = tmp_path / "models" / "vdn"
    ckpt = root / "stage"
    module = "token_refiner.refiner_blocks.0.attn.to_q"
    adapter = _adapter_entry(module) if with_adapter else None
    spec = _model_spec(adapter=adapter)
    _write_json(ckpt / "model_spec.json", spec)
    _write_json(ckpt / "linear_branch" / "config.json", spec["transforms"][0])
    (ckpt / "linear_branch" / "model.safetensors").touch()
    states = {str(ckpt / "linear_branch" / "model.safetensors"): _branch_state()}
    if adapter:
        adir = ckpt / "adapters" / "default"
        _write_json(adir / "adapter_config.json", adapter)
        (adir / "adapter_model.safetensors").touch()
        states[str(adir / "adapter_model.safetensors")] = {
            f"{module}.lora_A.default.weight": torch.zeros(1, 4),
            f"{module}.lora_B.default.weight": torch.zeros(4, 1),
        }
    return root, ckpt, states


def test_transform_and_model_spec_validation_are_resolved_and_typed():
    spec = _model_spec()
    validate_model_spec(spec)
    cfg = transform_config(spec)
    assert cfg["short_conv"] == ("k", "v")
    assert cfg["linear_head_dim"] == 2

    spec["transforms"][0]["config"]["linear_attention"]["linear_head_dim"] = None
    with pytest.raises(ValueError, match="unresolved"):
        validate_model_spec(spec)


def test_transform_rejects_declared_but_unimplemented_delta_rule():
    spec = _model_spec()
    spec["transforms"][0]["config"]["linear_attention"]["delta_rule"] = "vdn_scaled"
    with pytest.raises(ValueError, match="unsupported delta_rule"):
        validate_model_spec(spec)


def test_realpath_resolution_refuses_parent_traversal(tmp_path):
    root, ckpt, _ = _checkpoint(tmp_path)
    assert resolve_vdn_checkpoint("stage", roots=[root]) == str(ckpt)
    with pytest.raises(ValueError, match="escapes models/vdn"):
        resolve_vdn_checkpoint("../outside", roots=[root])


def test_inventory_rejects_non_safetensors_weight_and_incomplete_adapter(tmp_path):
    root, ckpt, _ = _checkpoint(tmp_path, with_adapter=False)
    (ckpt / "weights.pt").touch()
    with pytest.raises(ValueError, match="non-safetensors"):
        inventory_checkpoint(ckpt, roots=[root])

    (ckpt / "weights.pt").unlink()
    (ckpt / "adapters" / "broken").mkdir(parents=True)
    with pytest.raises(ValueError, match="no adapter directories are skipped"):
        inventory_checkpoint(ckpt, roots=[root])


def test_load_validates_complete_branch_and_adapter_inventory(tmp_path):
    root, ckpt, states = _checkpoint(tmp_path)

    def loader(path):
        return states[path]

    loaded = load_vdn_checkpoint(ckpt, roots=[root], tensor_loader=loader)
    assert len(loaded.branches) == 1
    assert set(loaded.adapters) == {"default"}
    assert "alpha.A_log" in loaded.branches[0]
    assert loaded.config["delta_rule"] == "vdn_solve"


def test_load_refuses_unknown_or_wrong_shape_branch_tensor(tmp_path):
    root, ckpt, states = _checkpoint(tmp_path, with_adapter=False)
    branch_path = str(ckpt / "linear_branch" / "model.safetensors")
    states[branch_path]["transformer_blocks.0.attn.unknown.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="inventory mismatch"):
        load_vdn_checkpoint(ckpt, roots=[root], tensor_loader=lambda path: states[path])

    states[branch_path].pop("transformer_blocks.0.attn.unknown.weight")
    states[branch_path]["transformer_blocks.0.attn.to_out_linear.weight"] = torch.zeros(5, 4)
    with pytest.raises(ValueError, match="hidden size"):
        load_vdn_checkpoint(ckpt, roots=[root], tensor_loader=lambda path: states[path])
