import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vdn_h3.bypass import CompositeQKVBypassAdapter, FrugalLoRABypassAdapter, _BypassHook


@pytest.mark.parametrize("native_first", [False, True])
def test_native_lora_bypass_coexists_in_fifo_lifecycle(monkeypatch, native_first):
    from comfy.weight_adapter.bypass import BypassForwardHook
    from comfy.weight_adapter.lora import LoRAAdapter
    import comfy.model_management

    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    torch.manual_seed(231)
    module = nn.Linear(4, 6, bias=False)
    native_up, native_down = torch.randn(6, 2), torch.randn(2, 4)
    vdn_up, vdn_down = torch.randn(6, 3), torch.randn(3, 4)
    native = BypassForwardHook(
        module, LoRAAdapter(set(), (native_up, native_down, None, None, None, None)),
    )
    vdn = _BypassHook(module, CompositeQKVBypassAdapter([
        (FrugalLoRABypassAdapter(vdn_up, vdn_down, 1.0), None),
    ]))
    x = torch.randn(5, 4)
    base = module(x)
    expected = base + F.linear(F.linear(x, native_down), native_up)
    expected += F.linear(F.linear(x, vdn_down), vdn_up)
    hooks = [native, vdn] if native_first else [vdn, native]
    for _ in range(2):
        for hook in hooks:
            hook.inject()
        torch.testing.assert_close(module(x), expected)
        for hook in hooks:
            hook.eject()
        torch.testing.assert_close(module(x), base)
        assert not module._forward_hooks


def test_linear_post_hook_preserves_keyword_input_and_repeated_injection():
    module = nn.Linear(2, 2, bias=False)
    adapter = CompositeQKVBypassAdapter([
        (FrugalLoRABypassAdapter(torch.ones(2, 1), torch.ones(1, 2), 1.0), None),
    ])
    hook = _BypassHook(module, adapter)
    x = torch.ones(1, 2)
    base = module(input=x)
    assert hook.inject() is True
    assert hook.inject() is False
    torch.testing.assert_close(module(input=x), base + 2)
    assert hook.eject() is True
    assert hook.eject() is False
    torch.testing.assert_close(module(input=x), base)
