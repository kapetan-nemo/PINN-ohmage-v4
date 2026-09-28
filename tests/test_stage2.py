"""Тесты этапа 2/6: метрики, пороги, прогон состояния, кодирующая сеть."""
import numpy as np
import torch

from pinn_soh.eval.levels import crossing_report, multi_cell_table
from pinn_soh.eval.metrics import (
    interval_coverage, rmse, soh_rmse_by_horizon, threshold_cycle,
    threshold_errors,
)
from pinn_soh.models.encoder import CellEncoder, Z_BOUNDS, Z_DIM, build_features
from pinn_soh.physics.degradation import DegradationConsts
from pinn_soh.train.stage2_degradation import (
    CellData, loss_rollout, rollout_state, trapz_weights,
)


def _cell(phi_level: float = 0.08, n_cyc: int = 20, n_pts: int = 50):
    """Синтетический CellData: φ_n ≈ const, ток ±1 мА·ч-циклы."""
    rng = np.random.default_rng(0)
    t = np.tile(np.linspace(0, 7200, n_pts), (n_cyc, 1))
    i = np.tile(
        np.where(np.arange(n_pts) < n_pts // 2, 1e-3, -1e-3), (n_cyc, 1))
    phi = np.full((n_cyc, n_pts), phi_level) + rng.normal(0, 5e-3, (n_cyc, n_pts))
    q0 = 0.003
    q_li = q0 * (1 - 5e-4 * np.arange(n_cyc))
    return CellData(
        cell_id="synth", cycles=np.arange(n_cyc, dtype=float),
        q_li=q_li, r_total=np.full(n_cyc, 0.05),
        t_s=torch.tensor(t), i_a=torch.tensor(i),
        phi_n=torch.tensor(phi), q_dch_meas=None,
        feats=torch.zeros(22, dtype=torch.float64))


def test_threshold_cycle_interpolation():
    c = np.array([0, 1, 2, 3.0])
    s = np.array([1.0, 0.85, 0.75, 0.6])
    assert abs(threshold_cycle(c, s, 0.8) - 1.5) < 1e-9
    assert threshold_cycle(c, s, 0.5) is None       # не достигнут
    assert threshold_cycle(c, s, 1.01) == 0.0       # ниже порога сразу


def test_threshold_errors_and_coverage():
    ct = np.arange(100.0); st = 1 - 0.002 * ct
    cp = np.arange(100.0); sp = 1 - 0.0025 * cp
    errs = threshold_errors(cp, sp, ct, st, levels=(0.9,))
    assert abs(errs[0.9] - (0.1 / 0.0025 - 0.1 / 0.002)) < 1
    assert interval_coverage(np.zeros(5), np.ones(5), np.full(5, 0.5)) == 1.0
    assert interval_coverage(np.zeros(5), np.ones(5), np.full(5, 2.0)) == 0.0


def test_soh_rmse_by_horizon():
    cp = np.arange(0, 600, 5.0); sp = 1 - 1e-4 * cp
    ct = np.arange(0, 1001, 1.0); st = 1 - 1.2e-4 * ct
    out = soh_rmse_by_horizon(cp, sp, ct, st, horizons=(100, 500))
    assert out[100] > 0 and out[500] > out[100]


def test_trapz_weights_exact():
    t = torch.tensor([[0., 1., 3., 4.]])
    w = trapz_weights(t)
    # ∫1 dt = 4
    assert abs(float((w * 1.0).sum()) - 4.0) < 1e-9


def test_rollout_monotone_and_differentiable():
    cell = _cell()
    enc = CellEncoder().double()
    z = enc(cell.feats)
    pred = rollout_state(cell, z, DegradationConsts())
    assert pred["q_li"].shape == cell.q_li.shape
    # при нулевом z и типичном φ_n запас лития убывает
    assert bool((pred["q_li"].diff() <= 1e-9).all())
    loss = loss_rollout(cell, pred)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in enc.parameters())


def test_encoder_bounds():
    enc = CellEncoder().double()
    f = torch.randn(7, 22, dtype=torch.float64)
    z = enc(f)
    assert z.shape == (7, Z_DIM)
    assert bool((z.abs() <= Z_BOUNDS + 1e-9).all())


def test_crossing_report_and_table():
    c = np.arange(100.0); s = 1 - 0.003 * c
    rep = crossing_report(c, s, levels=(0.9, 0.5))
    assert rep["0.9"]["reached"] and abs(rep["0.9"]["cross_cycle"] - 33.33) < 0.1
    assert not rep["0.5"]["reached"] and rep["0.5"]["rul_cycles"] is None
    tab = multi_cell_table([{"cell_id": "x", "cycles": c, "soh": s}],
                           levels=(0.9, 0.5))
    assert tab[0]["cell_id"] == "x" and len(tab) == 2


def test_build_features_dim():
    meta = {"v_max_main": 4.2, "q_n_ah": 0.003, "rho": 2.0,
            "theta_n0": [0.1, 0.1], "theta_p0": [0.9, 0.9],
            "q_li_ah": [0.003, 0.0029, 0.0028], "r_total_ohm": [0.04, 0.041],
            "c_p": [0.0] * 4, "c_n": [0.0] * 4}
    f = build_features(meta)
    assert f.shape == (22,) and bool(torch.isfinite(f).all())
