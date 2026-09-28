"""Кодирующая сеть элемента: признаки первых циклов → латентные множители z.

Выход ``z`` — логарифмические (по основанию 10) множители к константам
кинетики деградации OKane2022 и неидентифицируемым масштабам:

* ``z[0] lg k_SEI`` — константа скорости роста SEI;
* ``z[1] lg D_solv`` — диффузия растворителя через SEI;
* ``z[2] lg j0_pl`` — обменный ток электроосаждения лития;
* ``z[3] lg rho_SEI`` — удельное сопротивление слоя;
* ``z[4] lg A_eff`` — эффективная реакционная площадь анода.

Химия не задаётся меткой: вход включает латентный код электрода ``c_p``
и физически извлечённые параметры (``Q_n, ρ, θ-окна, R``), через которые
различие химий выражается в измерениях.
"""
from __future__ import annotations

import torch
import torch.nn as nn

Z_NAMES = ("lg_k_sei", "lg_d_solv", "lg_j0_pl", "lg_rho_sei", "lg_area_eff")
Z_DIM = len(Z_NAMES)

# диапазоны ограничения z (в декадах): кинетические множители ±3,
# площадь — ±6 (абсолютная шкала реакционной поверхности неизвестна)
Z_BOUNDS = torch.tensor([3.0, 3.0, 3.0, 3.0, 6.0], dtype=torch.float64)

LATENT_DIM = 4

# состав признаков элемента (порядок фиксирован)
FEATURE_NAMES = (
    "v_max_main",        # верхняя отсечка основного цикла, В /5
    "q_n_ah",            # идентифицированная ёмкость анода ×1000/4
    "rho",               # Q_n/Q_p /3
    "theta_n0_first",    # θ_n в начале первого идентифицированного цикла
    "theta_p0_first",
    "r0_mohm",           # R_total первого цикла, мОм /100
    "q_li_rel_0", "q_li_rel_1", "q_li_rel_2",  # q_li/q_li[0] циклов 0..2
    "r_rel_1", "r_rel_2",                      # r[k]/r[0]
    "dq_li_slope",       # относительный наклон q_li по первым 5 точкам ×1000
    "dr_slope",          # относительный наклон r ×1000
    "anchor_dist",       # расстояние кода катода до якоря (согласованность)
    *(f"c_p_{i}" for i in range(LATENT_DIM)),  # латентный код катода
    *(f"c_n_{i}" for i in range(LATENT_DIM)),  # латентный код анода
)
FEATURE_DIM = len(FEATURE_NAMES)


class CellEncoder(nn.Module):
    """MLP: признаки элемента → z (ограничен tanh·Z_BOUNDS)."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 64,
                 z_dim: int = Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, z_dim),
        )
        self.register_buffer("bounds", Z_BOUNDS[:z_dim].clone())
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.bounds * torch.tanh(self.net(feats.double()))


def build_features(meta: dict, anchor_code: torch.Tensor | None = None) -> torch.Tensor:
    """Вектор признаков из словаря идентифицированных параметров элемента.

    ``meta`` — объединённый словарь: контрольная запись качества
    (``v_max_main``) и контрольная точка идентификации. ``anchor_code`` —
    латентный код выбранного якоря для признака согласованности.
    """
    q_li = torch.tensor(meta["q_li_ah"], dtype=torch.float64)
    r = torch.tensor(meta["r_total_ohm"], dtype=torch.float64)
    tn = torch.tensor(meta["theta_n0"], dtype=torch.float64)
    tp = torch.tensor(meta["theta_p0"], dtype=torch.float64)

    def rel(x: torch.Tensor, k: int) -> torch.Tensor:
        return x[min(k, len(x) - 1)] / x[0]

    def slope(x: torch.Tensor, n: int = 5) -> torch.Tensor:
        m = min(n, len(x)) - 1
        if m < 1:
            return torch.zeros((), dtype=torch.float64)
        return (x[m] - x[0]) / (m * x[0]) * 1000.0

    c_p = torch.tensor(meta["c_p"][:LATENT_DIM], dtype=torch.float64)
    c_n = torch.tensor(meta["c_n"][:LATENT_DIM], dtype=torch.float64)
    a_dist = (c_p - anchor_code.to(torch.float64)).norm() if anchor_code is not None \
        else torch.zeros((), dtype=torch.float64)
    vals = [
        (meta.get("v_max_main") or 4.2) / 5.0,
        meta["q_n_ah"] * 1000.0 / 4.0,
        meta["rho"] / 3.0,
        tn[0], tp[0],
        r[0] * 1000.0 / 100.0,
        rel(q_li, 0), rel(q_li, 1), rel(q_li, 2),
        rel(r, 1), rel(r, 2),
        slope(q_li), slope(r),
        a_dist,
        *c_p, *c_n,
    ]
    return torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in vals])
