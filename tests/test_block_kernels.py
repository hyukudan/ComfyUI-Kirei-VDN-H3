from types import SimpleNamespace

import torch

from vdn_h3.block_kernels import make_fast_block_forward


def test_block_replace_attention_override_uses_native_contract():
    calls = []

    def native(x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        calls.append((t_emb, mod_segments, rope_freqs, transformer_options, attention))
        return attention(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    state = SimpleNamespace(block_fusion=True, inference=True, weight_mode="resident")
    forward = make_fast_block_forward(SimpleNamespace(forward=native), state, 2)
    args = {
        "x": torch.ones(3, 4), "t_emb": torch.randn(1, 4),
        "mod_segments": [(0, 3, 0)], "rope_freqs": object(),
        "transformer_options": {"patches": {"double_block": []}},
        "attention": lambda x, **kwargs: x * 7,
    }
    torch.testing.assert_close(forward(**args), args["x"] * 7)
    assert calls == [(args["t_emb"], args["mod_segments"], args["rope_freqs"],
                      args["transformer_options"], args["attention"])]
    assert state.block_fusion


def test_native_fallback_receives_none_attention():
    seen = []

    def native(x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        seen.append((transformer_options, attention))
        return x

    forward = make_fast_block_forward(SimpleNamespace(forward=native), SimpleNamespace(), 0)
    x = torch.ones(2, 4)
    assert forward(x, None, [], None) is x
    assert seen == [({}, None)]
