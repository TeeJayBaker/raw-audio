from __future__ import annotations

import torch
from torch import nn

from optim import build_muon
from optim.muon import SingleDeviceMuonWithAuxAdam, zeropower_via_newtonschulz5


def test_newtonschulz_orthogonalizes():
    torch.manual_seed(0)
    g = torch.randn(32, 64)
    s_in = torch.linalg.svdvals(g)
    o = zeropower_via_newtonschulz5(g, 5).float()
    s_out = torch.linalg.svdvals(o)
    # the iteration drives singular values toward ~Uniform(0.5, 1.5): better conditioned, bounded
    assert (s_out.max() / s_out.min()).item() < (s_in.max() / s_in.min()).item()
    assert s_out.max().item() < 2.0 and s_out.min().item() > 0.3
    assert o.shape == g.shape


def test_newtonschulz_handles_tall_matrices():
    torch.manual_seed(0)
    g = torch.randn(64, 16)  # rows > cols -> internal transpose path
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(g, 5).float())
    assert s.max().item() < 2.0 and s.min().item() > 0.3


def test_build_muon_groups_by_ndim():
    model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 8))
    opt = build_muon(model.parameters(), lr=0.02, adamw_lr=1e-3)
    assert isinstance(opt, SingleDeviceMuonWithAuxAdam)
    muon = [p for grp in opt.param_groups if grp["use_muon"] for p in grp["params"]]
    aux = [p for grp in opt.param_groups if not grp["use_muon"] for p in grp["params"]]
    assert all(p.ndim >= 2 for p in muon) and len(muon) == 2  # two Linear weights
    assert all(p.ndim < 2 for p in aux) and len(aux) == 4  # two biases + LayerNorm weight/bias


def test_build_muon_reduces_loss():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 8))
    opt = build_muon(model.parameters(), lr=0.02, adamw_lr=1e-3, weight_decay=0.01)
    x, y = torch.randn(16, 8), torch.randn(16, 8)
    first = loss = None
    for _ in range(100):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert torch.isfinite(loss) and loss.item() < 0.5 * first


def test_build_muon_accepts_generator_and_steps():
    model = nn.Linear(8, 8)
    opt = build_muon(model.parameters(), lr=0.02)  # bare generator input
    opt.zero_grad()
    model(torch.randn(2, 8)).sum().backward()
    opt.step()  # must not raise
