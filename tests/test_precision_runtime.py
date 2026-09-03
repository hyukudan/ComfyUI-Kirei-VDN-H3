from types import SimpleNamespace

import torch

from vdn_h3.block_kernels import BlockPointwiseCache
from vdn_h3.projection import detect_base_precision
from vdn_h3.weights import FP8_STREAMED_PROJECTION_KEY, ManagedBranchWeights
from vdn_h3.window import window_softmax_flash2


class _ModelPatcher:
    def __init__(self, layout=None, dtype=None):
        weight = SimpleNamespace(_layout_cls=layout, dtype=dtype)
        qkv = SimpleNamespace(weight=weight)
        attn = SimpleNamespace(qkv_proj=qkv)
        block = SimpleNamespace(attn=attn)
        self.dm = SimpleNamespace(blocks=[block])

    def get_model_object(self, key):
        assert key == "diffusion_model"
        return self.dm


def test_detect_base_precision_follows_comfy_quantized_layout():
    assert detect_base_precision(_ModelPatcher("TensorWiseINT8Layout")) == "int8"
    assert detect_base_precision(_ModelPatcher(None, torch.bfloat16)) == "bf16"


def test_int8_quantized_projection_storage_preserves_dtype():
    block = {
        FP8_STREAMED_PROJECTION_KEY: torch.zeros(8, 8, dtype=torch.int8),
        "to_out_linear.weight_scale": torch.ones(1, 8, dtype=torch.float32),
        "small": torch.ones(4, dtype=torch.bfloat16),
    }
    store = ManagedBranchWeights(
        [block],
        mode="hybrid",
        pin_strategy="none",
        streamed_keys=(FP8_STREAMED_PROJECTION_KEY,),
    )
    got = store.weights_on(0, "cpu", torch.bfloat16)
    assert got[FP8_STREAMED_PROJECTION_KEY].dtype == torch.int8
    assert got["to_out_linear.weight_scale"].dtype == torch.float32


def test_block_modulation_indices_accept_scalar_and_per_token_rows():
    cache = BlockPointwiseCache()
    scalar = cache.indices([(0, 2, 0), (2, 5, 2)], 5, torch.device("cpu"))
    assert scalar.tolist() == [0, 0, 2, 2, 2]

    vector = cache.indices(
        [(0, 2, 1), (2, 5, torch.tensor([0, 2, 1], dtype=torch.long))],
        5,
        torch.device("cpu"),
    )
    assert vector.tolist() == [1, 1, 0, 2, 1]


def test_flash2_backend_symbol_is_always_exported():
    # Regression guard for the intermediate import failure caught during the real
    # ComfyUI smoke test: hybrid.py must never reference a backend missing here.
    assert callable(window_softmax_flash2)
