import pytest
import torch

from vdn_h3_private.adapters import (
    convert_adapter,
    map_lora_target,
    parse_adapter_state,
    patches_by_target,
)


def _key(module, side, name="default"):
    return f"{module}.lora_{side}.{name}.weight"


def _spec(targets, rank=2, alpha=2, **extra):
    config = {"rank": rank, "alpha": alpha, "targets": targets, **extra}
    return {"type": "lora", "version": 1, "config": config}


def test_token_refiner_attention_maps_to_native_fused_qkv():
    target = map_lora_target("token_refiner.refiner_blocks.1.attn.to_k")
    assert target.key == "token_refiner.blocks.1.attn.qkv_proj"
    assert target.qkv_slice == "k"
    assert map_lora_target(
        "token_refiner.refiner_blocks.0.attn.to_out.0"
    ).key == "token_refiner.blocks.0.attn.out_proj"


def test_parse_is_a_strict_inventory():
    module = "transformer_blocks.0.attn.orig.to_q"
    spec = _spec([module], exact_targets=True)
    with pytest.raises(ValueError, match="exactly lora_A and lora_B"):
        parse_adapter_state({_key(module, "A"): torch.zeros(2, 3)}, spec)
    with pytest.raises(ValueError, match="no tensor is skipped"):
        parse_adapter_state({"metadata": torch.zeros(1)}, spec)
    with pytest.raises(ValueError, match="tensor rank"):
        parse_adapter_state(
            {_key(module, "A"): torch.zeros(1, 3), _key(module, "B"): torch.zeros(4, 1)},
            spec,
        )


def test_qkv_conversion_uses_three_compact_offset_deltas():
    modules = [f"transformer_blocks.0.attn.orig.to_{part}" for part in "qkv"]
    state = {}
    expected = []
    for index, module in enumerate(modules, 1):
        a = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        b = torch.full((4, 2), float(index))
        state[_key(module, "A")] = a
        state[_key(module, "B")] = b
        expected.append(b @ a)
    patches = convert_adapter(
        state,
        _spec(modules),
        target_shapes={"blocks.0.attn.qkv_proj.weight": (12, 3)},
    )
    assert [patch.offset for patch in patches] == [(0, 0), (4, 0), (8, 0)]
    assert all(patch.key == "blocks.0.attn.qkv_proj.weight" for patch in patches)
    assert sum(patch.delta.numel() for patch in patches) == 12 * 3
    for patch, want in zip(patches, expected):
        torch.testing.assert_close(patch.delta, want)
    assert len(patches_by_target(patches)[patches[0].key]) == 3


def test_mixed_rank_and_alpha_patterns_scale_each_projection_exactly():
    q = "transformer_blocks.0.attn.orig.to_q"
    k = "transformer_blocks.0.attn.orig.to_k"
    state = {
        _key(q, "A"): torch.ones(1, 2), _key(q, "B"): torch.ones(2, 1),
        _key(k, "A"): torch.ones(2, 2), _key(k, "B"): torch.ones(2, 2),
    }
    spec = _spec(
        [q, k],
        rank=1,
        alpha=1,
        rank_pattern={k: 2},
        alpha_pattern={k: 1},
        exact_targets=True,
    )
    q_patch, k_patch = convert_adapter(state, spec)
    torch.testing.assert_close(q_patch.delta, torch.ones(2, 2))
    # B@A == 2; alpha/rank == 1/2, so the exact delta is one.
    torch.testing.assert_close(k_patch.delta, torch.ones(2, 2))


def test_swiglu_b_rows_are_changed_from_value_gate_to_gate_value():
    module = "transformer_blocks.2.ff.net.0.proj"
    a = torch.eye(2)
    # Upstream rows are [value (10,11); gate (20,21)].
    b = torch.tensor([[10.0, 0.0], [11.0, 0.0], [20.0, 0.0], [21.0, 0.0]])
    state = {_key(module, "A"): a, _key(module, "B"): b}
    (patch,) = convert_adapter(
        state,
        _spec([module], exact_targets=True),
        target_shapes={"blocks.2.mlp.fc1.weight": (4, 2)},
    )
    torch.testing.assert_close(patch.delta, torch.cat((b[2:], b[:2])) @ a)
    assert patch.offset == (0, 0)


def test_conversion_validates_native_shapes_and_rejects_odd_swiglu():
    q = "token_refiner.refiner_blocks.0.attn.to_q"
    state = {_key(q, "A"): torch.zeros(2, 3), _key(q, "B"): torch.zeros(4, 2)}
    with pytest.raises(ValueError, match="incompatible with native fused target"):
        convert_adapter(state, _spec([q]), target_shapes={"token_refiner.blocks.0.attn.qkv_proj.weight": (15, 3)})

    ff = "transformer_blocks.0.ff.net.0.proj"
    odd = {_key(ff, "A"): torch.zeros(2, 3), _key(ff, "B"): torch.zeros(5, 2)}
    with pytest.raises(ValueError, match="not even"):
        convert_adapter(odd, _spec([ff]))

