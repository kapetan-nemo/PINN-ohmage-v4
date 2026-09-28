"""Медленный масштаб: кинетика деградации и обновление состояния цикла.

По окончании каждого смоделированного цикла по траекториям потенциала
анода ``φ_n(t)``, тока ``I(t)`` и температуры вычисляются интегралы
паразитных токов и обновляется медленное состояние:

* рост пассивирующего межфазного слоя (SEI) — диффузионно-кинетическая
  форма Пизона–Базанта: кинетический предел задаётся экспонентой от
  ``φ_n − U_SEI``, диффузионный — толщиной слоя δ_SEI; при накоплении слоя
  скорость переходит в режим ~1/δ, что даёт закон ~√t как следствие
  уравнений;
* катодное электроосаждение металлического лития при ``φ_n < 0``
  (уравнение Батлера–Фольмера для реакции осаждения, вклад части
  осаждённого лития в необратимую потерю — «мёртвый литий»);
* потеря запаса циклируемого лития (LLI) = заряд паразитных реакций,
  отнесённый к опорной ёмкости;
* рост ``R_total`` через сопротивление слоя: ``R_SEI = δ_SEI·ρ_SEI/A``;
* малая ограниченная остаточная поправка ``h_ψ`` для механизмов, не
  представленных явно (потеря активной массы и др.) — обучаемый член.

Литературные константы — набор OKane2022 (``data/params/kinetics_*.json``).
Обучаемые параметры элемента — логарифмические множители
``z = (lg k_SEI, lg D_solv, lg j0_pl, lg κ_SEI)`` — выход кодирующей сети;
остальные константы фиксированы.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

F_CONST = 96485.33212
R_CONST = 8.314462618
T_REF = 298.15


@dataclass
class DegradationConsts:
    """Литературные константы кинетики деградации (OKane2022)."""

    c_solv_mol_m3: float = 2636.0        # объёмная концентрация растворителя
    d_solv_m2_s: float = 2.5e-22         # эффективная диффузия через SEI
    k_sei_m_s: float = 1e-12             # константа скорости реакции SEI
    u_sei_v: float = 0.4                 # равновесный потенциал SEI vs Li/Li+
    v_sei_m3_mol: float = 9.585e-5       # парциальный молярный объём SEI
    rho_sei_ohm_m: float = 2e5           # удельное сопротивление SEI
    alpha_sei: float = 0.5               # коэффициент переноса SEI
    j0_pl_a_m2: float = 1e-3             # плотность обменного тока осаждения Li
    alpha_pl: float = 0.65               # коэффициент переноса осаждения
    beta_dead: float = 0.5               # доля осаждённого Li → мёртвый литий
    delta_sei0_m: float = 5e-9           # начальная толщина SEI
    k_dead_li_s: float = 1e-6            # константа распада мёртвого лития

    @classmethod
    def from_json(cls, path: str | Path, param_set: str | None = None) -> "DegradationConsts":
        """Загружает константы из выгрузки PyBaMM (``kinetics_*.json``)."""
        doc = json.loads(Path(path).read_text())
        consts = doc.get("constants", doc)
        def val(*names, default=None):
            for n in names:
                for k, v in consts.items():
                    if k.lower() == n.lower():
                        return v["value"] if isinstance(v, dict) else v
            return default
        return cls(
            c_solv_mol_m3=val("Bulk solvent concentration [mol.m-3]", default=cls.c_solv_mol_m3),
            d_solv_m2_s=val("Outer SEI solvent diffusivity [m2.s-1]",
                            "SEI solvent diffusivity [m2.s-1]", default=cls.d_solv_m2_s),
            k_sei_m_s=val("SEI kinetic rate constant [m.s-1]",
                          "SEI growth rate constant [m.s-1]", default=cls.k_sei_m_s),
            u_sei_v=val("SEI open-circuit potential [V]",
                        "Inner SEI open-circuit potential [V]", default=cls.u_sei_v),
            v_sei_m3_mol=val("SEI partial molar volume [m3.mol-1]",
                             "SEI molar volume [m3.mol-1]", default=cls.v_sei_m3_mol),
            rho_sei_ohm_m=val("SEI resistivity [Ohm.m]", default=cls.rho_sei_ohm_m),
            alpha_pl=val("Lithium plating transfer coefficient", default=cls.alpha_pl),
            k_dead_li_s=val("Dead lithium decay constant [s-1]",
                            "Dead lithium decay rate [s-1]", default=cls.k_dead_li_s),
            delta_sei0_m=val("Outer SEI thickness [m]", "Initial SEI thickness [m]",
                             default=cls.delta_sei0_m),
        )


@dataclass
class SlowState:
    """Медленное состояние элемента на границе циклов."""

    q_li_ah: float        # запас циклируемого лития в единицах ёмкости, А·ч
    delta_sei_m: float    # толщина SEI, м
    r_ohm0: float         # омическое сопротивление без SEI, Ом
    r_total_ohm: float    # R_Ω + R_SEI, Ом
    n_dead_ah: float = 0.0  # накопленный мёртвый литий, А·ч


@dataclass
class DegradationMultipliers:
    """Логарифмические множители z (выход кодирующей сети), по основанию 10."""

    lg_k_sei: float = 0.0
    lg_d_solv: float = 0.0
    lg_j0_pl: float = 0.0
    lg_rho_sei: float = 0.0


def sei_current_density(
    phi_n: torch.Tensor,
    delta_sei_m: torch.Tensor | float,
    consts: DegradationConsts,
    mult: DegradationMultipliers | None = None,
    temp_k: float = T_REF,
) -> torch.Tensor:
    """Плотность тока роста SEI, А/м² (отрицательна — катодный процесс).

    ``j_SEI = −F·c_solv / ( δ/D_solv + (1/k_SEI)·exp(αF/(RT)·(φ_n−U_SEI)) )``
    """
    m = mult or DegradationMultipliers()
    d_solv = consts.d_solv_m2_s * (10.0 ** m.lg_d_solv)
    k_sei = consts.k_sei_m_s * (10.0 ** m.lg_k_sei)
    expon = torch.exp(
        consts.alpha_sei * F_CONST / (R_CONST * temp_k) * (phi_n - consts.u_sei_v)
    ).clamp(max=1e6)
    denom = delta_sei_m / d_solv + expon / k_sei
    return -F_CONST * consts.c_solv_mol_m3 / denom


def plating_current_density(
    phi_n: torch.Tensor,
    consts: DegradationConsts,
    mult: DegradationMultipliers | None = None,
    temp_k: float = T_REF,
) -> torch.Tensor:
    """Плотность тока электроосаждения Li, А/м² (ненулевой при φ_n < 0).

    ``j_pl = j0_pl·[exp(αFφ_n/RT) − exp(−αFφ_n/RT)]·1[φ_n < 0]``
    """
    m = mult or DegradationMultipliers()
    j0 = consts.j0_pl_a_m2 * (10.0 ** m.lg_j0_pl)
    a = consts.alpha_pl * F_CONST / (R_CONST * temp_k)
    expo = j0 * (torch.exp((a * phi_n).clamp(max=50.0))
                 - torch.exp((-a * phi_n).clamp(max=50.0)))
    return torch.where(phi_n < 0.0, expo, torch.zeros_like(expo))


@dataclass
class CycleDegradation:
    """Интегралы паразитных токов за цикл и приращения состояния."""

    int_sei_a_s_m2: float = 0.0   # ∫|j_SEI| dt, А·с/м²
    int_pl_a_s_m2: float = 0.0    # ∫|j_pl| dt, А·с/м²
    dq_lli_ah: float = 0.0        # потеря лития за цикл, А·ч
    ddelta_sei_m: float = 0.0     # прирост толщины SEI, м
    dq_dead_ah: float = 0.0       # прирост мёртвого лития, А·ч


def cycle_degradation(
    t_s: torch.Tensor,
    i_a: torch.Tensor,
    phi_n: torch.Tensor,
    state: SlowState,
    consts: DegradationConsts,
    mult: DegradationMultipliers | None = None,
    area_eff_m2: float = 1.0,
    temp_k: float = T_REF,
    residual_fn=None,
) -> CycleDegradation:
    """Интегрирует кинетику деградации по траектории одного цикла.

    ``area_eff_m2`` — эффективная площадь реакционной поверхности анода
    (произведение удельной площади на объём/толщину электрода); по умолчанию
    1 м² — плотности тока в этом случае трактуются как полные токи.
    """
    j_sei = sei_current_density(phi_n, state.delta_sei_m, consts, mult, temp_k)
    j_pl = plating_current_density(phi_n, consts, mult, temp_k)
    n = t_s.numel()
    # веса трапеций: w[k] = (Δt_{k→k+1} + Δt_{k−1→k})/2, концы — половины
    dt = torch.diff(t_s)
    w = torch.zeros(n, dtype=torch.float64)
    if n > 1:
        w[0] = 0.5 * dt[0]
        w[-1] = 0.5 * dt[-1]
        if n > 2:
            w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    int_sei = float((j_sei.abs() * w).sum()) * area_eff_m2 / 3600.0  # А·ч экв.
    int_pl = float((j_pl.abs() * w).sum()) * area_eff_m2 / 3600.0
    dq_dead = consts.beta_dead * int_pl
    dq_lli = int_sei + dq_dead
    if residual_fn is not None:
        dq_lli = dq_lli + float(residual_fn(t_s, i_a, phi_n, state))
    ddelta = int_sei * 3600.0 * consts.v_sei_m3_mol / (F_CONST * area_eff_m2)
    return CycleDegradation(
        int_sei_a_s_m2=int_sei * 3600.0 / max(area_eff_m2, 1e-12),
        int_pl_a_s_m2=int_pl * 3600.0 / max(area_eff_m2, 1e-12),
        dq_lli_ah=dq_lli,
        ddelta_sei_m=ddelta,
        dq_dead_ah=dq_dead,
    )


def apply_cycle_update(
    state: SlowState,
    deg: CycleDegradation,
    consts: DegradationConsts,
    mult: DegradationMultipliers | None = None,
    area_m2: float = 1.0,
    q_li0_ah: float | None = None,
) -> SlowState:
    """Обновляет медленное состояние по приращениям цикла.

    ``n_Li`` убывает на величину потерянного заряда; толщина SEI растёт;
    полное сопротивление обновляется как ``R_Ω0 + δ_SEI·ρ_SEI/A_geo``.
    Монотонность обеспечена структурой: все приращения неотрицательны.
    """
    m = mult or DegradationMultipliers()
    rho_sei = consts.rho_sei_ohm_m * (10.0 ** m.lg_rho_sei)
    q_li_new = max(state.q_li_ah - deg.dq_lli_ah, 1e-6)
    delta_new = max(state.delta_sei_m + deg.ddelta_sei_m, 0.0)
    r_total = state.r_ohm0 + delta_new * rho_sei / area_m2
    return SlowState(
        q_li_ah=q_li_new,
        delta_sei_m=delta_new,
        r_ohm0=state.r_ohm0,
        r_total_ohm=r_total,
        n_dead_ah=state.n_dead_ah + deg.dq_dead_ah,
    )


class ResidualDegradation(nn.Module):
    """Ограниченная остаточная поправка к потере лития за цикл.

    Вход — интегральные характеристики цикла (полный заряд, средний и
    минимальный φ_n, доля времени φ_n<0, приращение δ_SEI) и латентный код
    элемента; выход — поправка к ΔQ_LLI, А·ч, ограниченная tanh и
    масштабом ``scale_ah``. Веса инициализируются нулями: до обучения
    поправка равна нулю (модель чисто механистическая).
    """

    def __init__(self, latent_dim: int = 3, hidden: int = 32, scale_ah: float = 1e-5):
        super().__init__()
        self.scale_ah = scale_ah
        self.net = nn.Sequential(
            nn.Linear(5 + latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, cycle_feats: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cycle_feats, c], dim=-1)
        return self.scale_ah * torch.tanh(self.net(x).squeeze(-1))


def resolve_stoich_windows(
    ocv_n,
    ocv_p,
    c: torch.Tensor,
    q_n_ah: float,
    rho: float,
    q_li_ah: float,
    v_min: float,
    v_max: float,
    n_grid: int = 400,
) -> dict[str, float]:
    """Стехиометрические окна по запасу лития и отсечкам напряжения.

    Баланс лития: ``θ_p = (L − θ_n·Q_n)/Q_p``; равновесное напряжение
    элемента ``V(θ_n) = U_p(θ_p) − U_n(θ_n)`` монотонно возрастает по θ_n.
    Бисекцией находятся θ_n на нижней (конец разряда) и верхней (конец
    заряда) отсечках; разрядная ёмкость ``Q_dch = (θ_n^max−θ_n^min)·Q_n``.
    """
    q_p_ah = q_n_ah / rho
    th = torch.linspace(1e-4, 0.9999, n_grid, dtype=torch.float64)
    theta_p = (q_li_ah - th * q_n_ah) / q_p_ah
    valid = (theta_p > 1e-4) & (theta_p < 0.9999)
    if valid.sum() < 8:
        return {"theta_n0": 0.5, "theta_p0": 0.5, "theta_n_chg": 0.5,
                "theta_p_chg": 0.5, "q_dch_ah": 0.0, "degenerate": True}
    th_v = th[valid]
    tp_v = theta_p[valid]
    ocv_cell = ocv_p(tp_v, c) - ocv_n(th_v, c)

    def solve(v_target: float) -> float:
        if v_target <= float(ocv_cell[0]):
            return float(th_v[0])
        if v_target >= float(ocv_cell[-1]):
            return float(th_v[-1])
        idx = int(torch.searchsorted(ocv_cell, v_target))
        x0, x1 = float(th_v[idx - 1]), float(th_v[idx])
        y0, y1 = float(ocv_cell[idx - 1]), float(ocv_cell[idx])
        w = (v_target - y0) / max(y1 - y0, 1e-12)
        return x0 + w * (x1 - x0)

    theta_n_min = solve(v_min)
    theta_n_max = solve(v_max)
    theta_p_min = (q_li_ah - theta_n_min * q_n_ah) / q_p_ah
    theta_p_max = (q_li_ah - theta_n_max * q_n_ah) / q_p_ah
    return {
        "theta_n0": theta_n_min,        # начало цикла (конец разряда)
        "theta_p0": theta_p_min,
        "theta_n_chg": theta_n_max,     # конец заряда по отсечке
        "theta_p_chg": theta_p_max,
        "q_dch_ah": (theta_n_max - theta_n_min) * q_n_ah,
        "degenerate": False,
    }
