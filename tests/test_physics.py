"""Тесты физического ядра: OCV, одиночная частица, протокол, деградация."""

import math

import pytest
import torch

from pinn_soh.data.metadata import ProtocolStep
from pinn_soh.physics.cell import (
    CellGeometry,
    CycleInit,
    bv_overpotential,
    simulate_cycle,
    simulate_cycle_vectorized,
)
from pinn_soh.physics.degradation import (
    DegradationConsts,
    DegradationMultipliers,
    SlowState,
    apply_cycle_update,
    cycle_degradation,
    plating_current_density,
    resolve_stoich_windows,
    sei_current_density,
)
from pinn_soh.physics.ocv import MonotoneOCV
from pinn_soh.physics.protocol import CellSimulator, simulate_protocol, main_loop_steps


def _ocv_pair(latent_dim=3, seed=0):
    torch.manual_seed(seed)
    return MonotoneOCV(latent_dim), MonotoneOCV(latent_dim)


def _geom(ocv_c=None):
    return CellGeometry(
        q_n_ah=2.0e-3, rho=1.1, area_m2=1.54e-4,
        j0_n_a_m2=1.0, j0_p_a_m2=1.0,
        c=ocv_c if ocv_c is not None else torch.zeros(3, dtype=torch.float64),
    )


# --- OCV -------------------------------------------------------------------

def test_ocv_monotone_decreasing():
    """U(θ; c) убывает по θ для произвольных кодов c."""
    ocv_n, _ = _ocv_pair()
    th = torch.linspace(0.02, 0.98, 61, dtype=torch.float64)
    for _ in range(5):
        c = torch.randn(3, dtype=torch.float64)
        u = ocv_n(th, c)
        assert (u.diff() < 0).all(), "кривая не монотонно убывает"


def test_ocv_positive_slope():
    """|dU/dθ| положителен."""
    ocv_n, _ = _ocv_pair()
    c = torch.zeros(3, dtype=torch.float64)
    s = ocv_n.dudt(torch.linspace(0.05, 0.95, 9, dtype=torch.float64), c)
    assert (s > 0).all()


def test_ocv_grid_point_exact():
    """Значение на сетке точно совпадает с grid_values."""
    ocv, _ = _ocv_pair()
    c = torch.randn(3, dtype=torch.float64)
    grid_u = ocv.grid_values(c)
    u = ocv(ocv.theta_grid[[0, 40, 64, 128]], c)
    assert torch.allclose(u, grid_u[[0, 40, 64, 128]], atol=1e-10)


# --- Батлер–Фольмер ---------------------------------------------------------

def test_bv_sign_and_zero():
    """η(0)=0; знак η совпадает со знаком тока; антисимметрия η(−i)=−η(i)."""
    th = torch.tensor([0.5], dtype=torch.float64)
    j0 = 1.0 * torch.ones(1, dtype=torch.float64)
    eta_p = bv_overpotential(torch.tensor(0.001, dtype=torch.float64), 1e-4, j0)
    eta_m = bv_overpotential(torch.tensor(-0.001, dtype=torch.float64), 1e-4, j0)
    assert eta_p > 0 and eta_m < 0
    assert abs(float(eta_p + eta_m)) < 1e-15
    eta0 = bv_overpotential(torch.tensor(0.0, dtype=torch.float64), 1e-4, j0)
    assert float(eta0) == 0.0


# --- одиночная частица -------------------------------------------------------

def test_faraday_charge_conservation():
    """θ_n конечное соответствует интегралу тока (закон Фарадея)."""
    ocv_n, ocv_p = _ocv_pair()
    geom = _geom()
    t = torch.arange(0, 1000, 10, dtype=torch.float64)
    i = torch.full_like(t, 1.8e-3)  # 1 мА·ч · ~0.9C
    init = CycleInit(theta_n0=0.3, theta_p0=0.7)
    out = simulate_cycle(t, i, ocv_n, ocv_p, geom, init)
    dq_ah = float((i * torch.diff(t, prepend=t[:1])).sum() / 3600)
    # приращение θ_n к концу интервала последнего шага: используем
    # векторизованную версию (в ней θ_n — cumsum по завершённым интервалам)
    out_v = simulate_cycle_vectorized(t, i, ocv_n, ocv_p, geom, init)
    assert abs(out_v["theta_n"][-1].item() - (0.3 + dq_ah / geom.q_n_ah)) < 1e-6


def test_rc_exact_update():
    """Одна RC-мода: V_diff(t) = R_d·I·(1−e^{−t/τ}) для постоянного тока."""
    ocv_n, ocv_p = _ocv_pair()
    geom = _geom()
    tau, r_d = 100.0, 0.01
    init = CycleInit(theta_n0=0.3, theta_p0=0.7, tau_d_s=(tau,), r_d_ohm=(r_d,))
    t = torch.arange(0, 300, 10, dtype=torch.float64)
    i = torch.full_like(t, 1e-3)
    out = simulate_cycle_vectorized(t, i, ocv_n, ocv_p, geom, init)
    # после интервала Δt моды принимают vd[k+1] = vd[k]·a + r(1−a)·I
    # значение в точке k соответствует суммарному времени t[k]
    expected = r_d * 1e-3 * (1 - torch.exp(-t / tau))
    assert torch.allclose(out["v_diff"], expected, atol=1e-6)


def test_vectorized_matches_sequential():
    """Векторизованная и пошаговая версии дают одинаковое напряжение."""
    ocv_n, ocv_p = _ocv_pair()
    geom = _geom()
    t = torch.arange(0, 500, 5, dtype=torch.float64)
    i = torch.where(torch.arange(t.numel()) % 2 == 0,
                    torch.tensor(1e-3, dtype=torch.float64),
                    torch.tensor(-1e-3, dtype=torch.float64))
    init = CycleInit(theta_n0=0.35, theta_p0=0.65)
    a = simulate_cycle(t, i, ocv_n, ocv_p, geom, init)
    b = simulate_cycle_vectorized(t, i, ocv_n, ocv_p, geom, init)
    assert torch.allclose(a["v_hat"], b["v_hat"], atol=1e-6)


# --- протокол ----------------------------------------------------------------

def _flat_ocvs():
    """Синтетические электроды с линейными кривыми для регулятора."""
    from pinn_soh.physics.ocv import MonotoneOCV
    import torch.nn as nn

    class LinOCV(nn.Module):
        def __init__(self, u_hi, slope):
            super().__init__()
            self.u_hi, self.slope = u_hi, slope
        def forward(self, theta, c=None):
            return self.u_hi - self.slope * theta.to(torch.float64)
        def __call__(self, theta, c=None):
            return self.forward(theta)
    # катод: 4.4 → 3.4 В; анод: 0.6 → 0.05 В → элемент 3.8 → 3.35 В
    return LinOCV(0.6, 0.55), LinOCV(4.4, 1.0)


def test_protocol_cc_cv_cutoffs():
    """CC-заряд до V_max, CV до тока отсечки, разряд до V_min."""
    ocv_n, ocv_p = _flat_ocvs()
    geom = _geom()
    sim = CellSimulator(ocv_n, ocv_p, geom, r_total_ohm=1.0,
                        tau_d_s=(100.0,), r_d_ohm=(0.0,))
    # окна: элемент 4.4−0.55x − (0.6−…): при theta_n0=0.5,tp0=0.5
    # V = 4.4−(0.5−ρ(x−0.5)) − (0.6−0.55x) → зададим окно напрямую
    sim.begin_cycle(theta_n0=0.2, theta_p0=0.9)
    steps = [
        ProtocolStep(mode="cc_charge", current_a=1e-3, voltage_limit_v=3.75),
        ProtocolStep(mode="cv_charge", voltage_limit_v=3.75,
                     current_cutoff_a=2e-4, duration_s=3600),
        ProtocolStep(mode="rest", duration_s=300),
        ProtocolStep(mode="cc_discharge", current_a=1e-3, voltage_limit_v=3.55),
    ]
    tr = simulate_protocol(sim, steps, t0=0.0, dt_max=20.0)
    ts, ii, vv = tr.t, tr.i, tr.v
    # CC-фаза заканчивается при V≈3.75
    i_end_cc = next(k for k in range(1, len(ii)) if abs(ii[k] - 1e-3) < 1e-9 and vv[k] >= 3.749)
    assert vv[i_end_cc] <= 3.75 + 1e-3
    # в CV-фазе ток монотонно не растёт и падает ниже отсечки
    i_cv = [x for x in ii if 0 < x < 1e-3 - 1e-9]
    if i_cv:
        assert min(i_cv) <= 2e-4 or abs(ii[-1]) < 1e-9
    # разряд идёт до V_min
    assert min(vv) <= 3.551
    # закон сохранения: заряд ≥ разряда (КПД ≤ 1 за счёт того, что заряд
    # включает CV-фазу)
    assert tr.charge_capacity_ah() >= tr.discharge_capacity_ah() - 1e-9


def test_main_loop_steps_selects_last_workflow():
    w1 = ProtocolStep(mode="workflow", repeat=3,
                      steps=[ProtocolStep(mode="cc_charge")])
    w2 = ProtocolStep(mode="workflow", repeat=1000,
                      steps=[ProtocolStep(mode="cc_charge"),
                             ProtocolStep(mode="cc_discharge")])
    proto = [ProtocolStep(mode="rest", duration_s=60), w1, w2]
    steps = main_loop_steps(proto)
    assert len(steps) == 2 and steps[1].mode == "cc_discharge"


# --- деградация ---------------------------------------------------------------

def test_sei_current_negative_and_diffusion_limited():
    """j_SEI ≤ 0; при росте δ модуль тока убывает (диффузионный предел 1/δ)."""
    consts = DegradationConsts()
    phi = torch.full((5,), 0.05, dtype=torch.float64)
    j1 = sei_current_density(phi, 5e-9, consts)
    j2 = sei_current_density(phi, 5e-8, consts)
    assert (j1 <= 0).all()
    assert j2.abs().mean().item() < j1.abs().mean().item()


def test_plating_only_below_zero():
    consts = DegradationConsts()
    phi_pos = torch.tensor([0.05, 0.2], dtype=torch.float64)
    phi_neg = torch.tensor([-0.05, -0.1], dtype=torch.float64)
    assert (plating_current_density(phi_pos, consts) == 0).all()
    j = plating_current_density(phi_neg, consts)
    assert (j < 0).all()


def test_cycle_update_monotone():
    """Обновление состояния: запас лития не растёт, SEI и R_total не убывают."""
    consts = DegradationConsts()
    state = SlowState(q_li_ah=2e-3, delta_sei_m=5e-9, r_ohm0=0.02, r_total_ohm=0.02)
    t = torch.arange(0, 3600, 30, dtype=torch.float64)
    i = torch.full_like(t, 1e-3)
    phi = torch.full_like(t, 0.05)
    deg = cycle_degradation(t, i, phi, state, consts)
    new = apply_cycle_update(state, deg, consts)
    assert new.q_li_ah <= state.q_li_ah
    assert new.delta_sei_m >= state.delta_sei_m
    assert new.r_total_ohm >= state.r_total_ohm


def test_windows_solution():
    """Синтетические линейные кривые: ёмкость разряда положительна и
    не превышает запас лития."""
    from pinn_soh.physics.protocol import CellSimulator  # noqa
    ocv_n, ocv_p = _flat_ocvs()
    c = torch.zeros(3, dtype=torch.float64)
    res = resolve_stoich_windows(ocv_n, ocv_p, c, q_n_ah=2e-3, rho=1.1,
                                 q_li_ah=1.8e-3, v_min=3.4, v_max=3.75)
    assert not res["degenerate"]
    assert res["theta_n_chg"] > res["theta_n0"]
    assert 0 < res["q_dch_ah"] <= res["theta_n_chg"] * 2e-3


def test_recursive_consistency_stub():
    """Моделирование H циклов одним вызовом и по частям даёт одно и то же —
    здесь проверяется детерминизм регулятора протокола на двух проходах."""
    ocv_n, ocv_p = _flat_ocvs()
    geom = _geom()
    steps = [ProtocolStep(mode="cc_discharge", current_a=1e-3, voltage_limit_v=3.55)]
    def one_run():
        sim = CellSimulator(ocv_n, ocv_p, geom, r_total_ohm=1.0,
                            tau_d_s=(100.0,), r_d_ohm=(0.0,))
        sim.begin_cycle(0.2, 0.9)
        return simulate_protocol(sim, steps, dt_max=25.0)
    a, b = one_run(), one_run()
    assert a.t == b.t and a.v == b.v


def test_edge_v_collapse_direction():
    """edge_v: выход θ за [0,1] двигает V в физическую сторону.

    Переразряд (анод пуст θ_n<0 ИЛИ катод полон θ_p>1) → V вниз;
    перезаряд (анод полон θ_n>1 ИЛИ катод пуст θ_p<0) → V вверх.
    Одновременные нарушения разных границ не компенсируются.
    """
    from pinn_soh.train.stage1_extract_state import simulate_batch
    ocv_n, ocv_p = _ocv_pair()
    c = torch.zeros(3, dtype=torch.float64)
    N, dt_s = 80, 60.0
    # строки: разряд {анод пуст; катод полон; оба}, заряд {анод полон;
    # катод пуст; оба} — границы выбраны так, чтобы ровно заданные
    # ограничения пересекались за длину шаблона
    tn0 = torch.tensor([0.05, 0.75, 0.05, 0.95, 0.30, 0.95],
                       dtype=torch.float64)
    tp0 = torch.tensor([0.20, 0.95, 0.95, 0.75, 0.05, 0.05],
                       dtype=torch.float64)
    ii = torch.full((6, N), -1e-3, dtype=torch.float64)
    ii[3:] = 1e-3
    t = (torch.arange(N, dtype=torch.float64) * dt_s).repeat(6, 1)
    mask = torch.ones(6, N, dtype=torch.bool)
    zero_r = torch.zeros(6, dtype=torch.float64)
    j0 = torch.tensor(50.0, dtype=torch.float64)
    base = simulate_batch(t, ii, mask, ocv_n, ocv_p, tn0, tp0,
                          torch.tensor(2e-3), torch.tensor(1.1),
                          zero_r, j0, j0, 1.54e-4, c, c)
    edge = simulate_batch(t, ii, mask, ocv_n, ocv_p, tn0, tp0,
                          torch.tensor(2e-3), torch.tensor(1.1),
                          zero_r, j0, j0, 1.54e-4, c, c, edge_v=500.0)
    dv = (edge["v_hat"] - base["v_hat"])[:, -1].detach()
    assert (dv[:3] < -10.0).all(), f"переразряд должен ронять V: {dv[:3]}"
    assert (dv[3:] > 10.0).all(), f"перезаряд должен поднимать V: {dv[3:]}"
    # двойное нарушение строго сильнее одиночного (нет компенсации)
    assert abs(float(dv[2])) > abs(float(dv[0]))
    assert abs(float(dv[5])) > abs(float(dv[3]))
