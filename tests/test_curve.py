import torch
import torch.nn as nn

from vdn_h3.adapters import FactorPatch
from vdn_h3.curve import (
    CurveAdapterState,
    curve_runtime_scope,
    is_curve_h3_base,
    make_curve_adaln_forward,
)


class TinyAdaln(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = 2
        self.modalities = 1
        self.hidden = 2
        self.linear = nn.Linear(2, 4, bias=False)

    def forward(self, value):
        flat = self.linear(value).view(value.shape[0], 4)
        return flat.chunk(2, dim=-1)


def test_curve_embedding_recovers_fractional_grid_coordinate():
    table = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    egrid = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [30.0, 40.0, 50.0]])
    patch = FactorPatch("target", torch.ones(4, 1), torch.ones(1, 3), None, "x", 1.0, True)
    state = CurveAdapterState(egrid, [(patch, 1.0)])
    coordinate = torch.lerp(table[1], table[2], 0.25).unsqueeze(0)
    with curve_runtime_scope():
        result = state.full_embedding(coordinate, table)
        again = state.full_embedding(coordinate, table)
    torch.testing.assert_close(result, torch.lerp(egrid[1], egrid[2], 0.25).unsqueeze(0))
    assert again.data_ptr() == result.data_ptr()


def test_curve_forward_adds_low_rank_delta_before_chunking():
    base = TinyAdaln()
    base.linear.weight.data.zero_()
    table = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    dm = nn.Module()
    dm.register_buffer("adaln_t_table", table)
    patch = FactorPatch(
        "target",
        torch.ones(4, 1),
        torch.tensor([[2.0, 0.0, 0.0]]),
        None,
        "x",
        0.5,
        True,
    )
    state = CurveAdapterState(torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), [(patch, 1.0)])
    forward = make_curve_adaln_forward(base, dm, state, "target")
    with curve_runtime_scope():
        first, second = forward(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(first, torch.full((1, 2), 2.0))
    torch.testing.assert_close(second, torch.full((1, 2), 2.0))


def test_curve_base_detection_uses_flag_or_collapsed_adaln_shape():
    flagged = nn.Module()
    flagged.use_adaln_curves = True
    assert is_curve_h3_base(flagged)

    structural = nn.Module()
    structural.blocks = nn.ModuleList([nn.Module()])
    structural.blocks[0].adaln_proj = nn.Module()
    structural.blocks[0].adaln_proj.linear = nn.Linear(8, 32, bias=False)
    assert is_curve_h3_base(structural)

    dense = nn.Module()
    dense.blocks = nn.ModuleList([nn.Module()])
    dense.blocks[0].adaln_proj = nn.Module()
    dense.blocks[0].adaln_proj.linear = nn.Linear(2688, 32, bias=False)
    assert not is_curve_h3_base(dense)
