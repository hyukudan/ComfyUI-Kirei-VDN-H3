import torch
import torch.nn as nn
import pytest
from safetensors.torch import save_file

from vdn_h3.adapters import FactorPatch
from vdn_h3.curve import (
    CurveAdapterState,
    curve_runtime_scope,
    is_curve_h3_base,
    make_curve_adaln_forward,
    load_egrid,
)
from vdn_h3 import curve


def test_affine_companion_recovers_matching_dense_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, '_egrid_candidates', lambda *args: iter(()))
    table = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    basis = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    mean = torch.tensor([0.1, 0.2, 0.3])
    save_file({'adaln_basis': basis, 'adaln_mean': mean, 'adaln_t_table': table}, str(tmp_path / 'adaln_affine.safetensors'))
    grid = load_egrid(checkpoint_root=str(tmp_path), table=table)
    torch.testing.assert_close(grid, table @ basis + mean)
    with pytest.raises(ValueError, match='different H3 curve table'):
        load_egrid(checkpoint_root=str(tmp_path), table=table + 1)


def test_egrid_remains_preferred_over_affine_companion(tmp_path, monkeypatch):
    path = tmp_path / 'h3_silu_temb_grid.safetensors'
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file({'silu_t_emb_grid': expected}, str(path))
    monkeypatch.setattr(curve, '_egrid_candidates', lambda *args: iter([path]))
    (tmp_path / 'adaln_affine.safetensors').write_bytes(b'invalid')
    torch.testing.assert_close(load_egrid(checkpoint_root=str(tmp_path), table=torch.zeros(2, 2)), expected)


def test_egrid_reports_source_and_validates_declared_base_identity(tmp_path):
    import hashlib

    path = tmp_path / 'h3_silu_temb_grid.safetensors'
    table = torch.zeros(2, 2)
    grid = torch.ones(2, 3)
    save_file({'silu_t_emb_grid': grid}, str(path))
    actual, source = curve._load_egrid(str(path), table=table)
    torch.testing.assert_close(actual, grid)
    assert source == {
        'kind': 'egrid', 'path': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'base_table_verified': False,
    }
    save_file({'silu_t_emb_grid': grid}, str(path), metadata={'adaln_table_sha256': curve._table_digest(table)})
    _, source = curve._load_egrid(str(path), table=table)
    assert source['base_table_verified'] is True
    with pytest.raises(ValueError, match='e-grid table hash'):
        load_egrid(str(path), table=table + 1)


@pytest.mark.parametrize('verified', [False, True])
def test_affine_source_reports_actual_identity_verification(tmp_path, monkeypatch, verified):
    monkeypatch.setattr(curve, '_egrid_candidates', lambda *args: iter(()))
    table = torch.zeros(2, 2)
    tensors = {'adaln_basis': torch.ones(2, 3), 'adaln_mean': torch.zeros(3)}
    if verified:
        tensors['adaln_t_table'] = table
    save_file(tensors, str(tmp_path / 'adaln_affine.safetensors'))
    _, source = curve._load_egrid(checkpoint_root=str(tmp_path), table=table)
    assert source['kind'] == 'affine'
    assert source['base_table_verified'] is verified


@pytest.mark.parametrize('metadata,basis', [({'adaln_table_sha256': 'wrong'}, torch.ones(2, 3)), ({}, torch.full((2, 3), float('nan')))])
def test_affine_companion_rejects_wrong_identity_or_invalid_values(tmp_path, monkeypatch, metadata, basis):
    monkeypatch.setattr(curve, '_egrid_candidates', lambda *args: iter(()))
    save_file({'adaln_basis': basis, 'adaln_mean': torch.zeros(3)}, str(tmp_path / 'adaln_affine.safetensors'), metadata=metadata)
    with pytest.raises(ValueError, match='table hash|non-finite'):
        load_egrid(checkpoint_root=str(tmp_path), table=torch.zeros(3, 2))


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
