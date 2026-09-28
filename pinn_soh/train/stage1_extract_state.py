"""Этап 1: идентификация состояния элемента по наблюдённым циклам.

Для элемента подгоняются:

* **постоянные параметры** (общие для всех циклов): латентные коды
  кривых равновесного потенциала ``c_n, c_p``, соотношение ёмкостей
  ``ρ = Q_n/Q_p``, абсолютная ёмкость анода ``Q_n``, масштабы обменных
  токов ``j0_n, j0_p``;
* **поцикловые параметры**: стехиометрии в начале цикла
  ``(θ_n0, θ_p0)``, полное сопротивление ``R_total``, (при наличии
  релаксаций — постоянные времени и амплитуды диффузионных мод).

Запас циклируемого лития выводится из идентифицированных окон:
``L_N = θ_n0·Q_n + θ_p0·Q_p`` (А·ч эквивалента); SOH-траектория для
сверки — из измеренной разрядной ёмкости.

Оптимизация: Adam в двух фазах — сначала поцикловые параметры при
замороженных константах (инициализация из якорей), затем совместная
тонкая подгонка на каждом ``stride``-м цикле; на последнем проходе —
интерполяция поцикловых параметров на пропущенные циклы.
Вычисления — float64 на центральном процессоре; циклы батчированы
тензорами (B, N) с маской.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch

from pinn_soh.physics.cell import F_CONST, R_CONST, T_REF
from pinn_soh.physics.degradation import resolve_stoich_windows
from pinn_soh.physics.ocv import MonotoneOCV, torchinterp1

N_POINTS = 300          # точек на цикл после прореживания/дополнения
TAU_D_DEFAULT = (60.0, 600.0)
R_D_DEFAULT = (0.005, 0.005)


# --- батчированная динамика -------------------------------------------------

def _u_eval(net: MonotoneOCV, th_flat: torch.Tensor, code: torch.Tensor,
            affine: tuple | None, delta: torch.Tensor | None) -> torch.Tensor:
    """U(th) с аффином, поправкой и линейной экстраполяцией за [0,1]."""
    tc = th_flat.clamp(0.0, 1.0)
    u = net(tc, code)
    if affine is not None:
        u = u * affine[0] + affine[1]
    if delta is not None:
        u = u + torchinterp1(net.theta_grid, delta, tc)
    s_a = affine[0] if affine is not None else 1.0
    z = torch.zeros(1, dtype=torch.float64)
    o = torch.ones(1, dtype=torch.float64)
    sl0 = net.dudt(z, code).squeeze() * s_a
    sl1 = net.dudt(o, code).squeeze() * s_a
    u = u + sl0 * torch.clamp(-th_flat, min=0.0) \
          - sl1 * torch.clamp(th_flat - 1.0, min=0.0)
    return u


def simulate_batch(
    t: torch.Tensor,      # (B,N) секунды
    i: torch.Tensor,      # (B,N) амперы, заряд положителен
    mask: torch.Tensor,   # (B,N) bool — действительные точки
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    theta_n0: torch.Tensor,   # (B,)
    theta_p0: torch.Tensor,   # (B,)
    q_n: torch.Tensor,        # скаляр
    rho: torch.Tensor,        # скаляр
    r_total: torch.Tensor,    # (B,) Ом
    j0n_scale: torch.Tensor,  # скаляр-множитель к j0_n
    j0p_scale: torch.Tensor,
    area_m2: float,
    c_n: torch.Tensor,        # (d,) латентные коды
    c_p: torch.Tensor,
    tau_d: tuple = TAU_D_DEFAULT,
    r_d: tuple = R_D_DEFAULT,
    temp_k: float = T_REF,
    ocv_n_affine: tuple | None = None,   # (scale, shift) коррекция кривой
    ocv_p_affine: tuple | None = None,
    ocv_p_delta: torch.Tensor | None = None,  # (G,) аддитивная коррекция U_p
    ocv_n_delta: torch.Tensor | None = None,
    hyst_v: torch.Tensor | None = None,       # (B,) полуразмах гистерезиса, В
    lam_n: torch.Tensor | None = None,  # (B,) относит. ёмкость анода цикла
    lam_p: torch.Tensor | None = None,  # (B,) относит. ёмкость катода цикла
    edge_v: float = 0.0,   # >0 — квадратичный обвал v за пределами θ∈[0,1]
) -> dict[str, torch.Tensor]:
    """Векторизованное моделирование пачки циклов (B,N).

    Возвращает ``v_hat, phi_n, theta_n, theta_p`` формы (B,N).
    """
    B, N = i.shape
    if lam_n is None:
        lam_n = torch.ones(B, dtype=torch.float64)
    if lam_p is None:
        lam_p = torch.ones(B, dtype=torch.float64)
    fwd = torch.diff(t, dim=1)                       # (B,N-1)
    dq = torch.cat(
        [torch.zeros(B, 1, dtype=torch.float64),
         torch.cumsum(i[:, :-1] * fwd, dim=1)], dim=1
    ) / (3600.0 * q_n)                               # доля Q_n_ref
    theta_n = theta_n0.unsqueeze(1) + dq / lam_n.unsqueeze(1)
    theta_p = theta_p0.unsqueeze(1) - rho * dq / lam_p.unsqueeze(1)

    def exch(j0_ref, th):
        thc = th.clamp(1e-4, 1 - 1e-4)
        return j0_ref * torch.sqrt(thc * (1 - thc))

    def bv(ii, j0):
        lim = (2.0 * area_m2 * j0).clamp_min(1e-12)
        return (2.0 * R_CONST * temp_k / F_CONST) * torch.asinh(ii / lim)

    eta_n = bv(i, exch(j0n_scale, theta_n))
    eta_p = bv(i, exch(j0p_scale, theta_p))

    # V_diff — точные RC-переходы, петля по времени без обращений к сети
    v_diff = torch.zeros(B, N, dtype=torch.float64)
    for tau, rd in zip(tau_d, r_d):
        vd = torch.zeros(B, dtype=torch.float64)
        acc = torch.empty(B, N, dtype=torch.float64)
        for k in range(N):
            if k > 0:
                a = torch.exp(-(t[:, k] - t[:, k - 1]) / tau)
                vd = vd * a + rd * (1 - a) * i[:, k - 1]
            acc[:, k] = vd
        v_diff = v_diff + acc

    u_p = _u_eval(ocv_p, theta_p.reshape(-1), c_p, ocv_p_affine,
                  ocv_p_delta).reshape(B, N)
    u_n = _u_eval(ocv_n, theta_n.reshape(-1), c_n, ocv_n_affine,
                  ocv_n_delta).reshape(B, N)
    v_hat = u_p - u_n + eta_p + eta_n + i * r_total.unsqueeze(1) + v_diff
    if edge_v > 0:
        # физический концевой обвал: за пределами стехиометрии
        # напряжение уходит в ±∞ — прогнозный разряд умирает на
        # границе окна инвентаря, а не за ним (в идентификации
        # выключено — там экстраполяция нужна оптимизатору).
        # Направление обвала зависит от электрода и стороны границы
        # (V = U_p − U_n, обе кривые убывают по θ):
        #   переразряд — катод заполнен θ_p>1 ИЛИ анод пуст θ_n<0 → V↓;
        #   перезаряд  — анод заполнен θ_n>1 ИЛИ катод пуст θ_p<0 → V↑.
        # Каждое нарушение — свой квадратичный член: односторонние
        # нарушения не компенсируют друг друга.
        over_dch = (theta_p - 1.0).clamp_min(0.0) ** 2 \
            + (-theta_n).clamp_min(0.0) ** 2
        over_chg = (theta_n - 1.0).clamp_min(0.0) ** 2 \
            + (-theta_p).clamp_min(0.0) ** 2
        v_hat = v_hat + edge_v * (over_chg - over_dch)
    if hyst_v is not None:
        # смещение ветви: заряд — вверх, разряд — вниз; гладко через I=0
        v_hat = v_hat + hyst_v.unsqueeze(1) * torch.tanh(i / 2e-4)
    phi_n = u_n - eta_n
    return {"v_hat": v_hat, "phi_n": phi_n, "theta_n": theta_n,
            "theta_p": theta_p, "eta_p": eta_p, "eta_n": eta_n,
            "u_p": u_p, "u_n": u_n, "v_diff": v_diff}


def pack_cycles(df: pl.DataFrame, cycle_ids: list[int], n_points: int = N_POINTS):
    """Собирает тензоры (B,N) по списку циклов из прореженного кадра."""
    t_list, i_list, v_list = [], [], []
    for cid in cycle_ids:
        sub = df.filter(pl.col("cycle") == cid).sort("t_s")
        n = sub.height
        tt = sub["t_s"].to_numpy()
        ii = sub["i_a"].to_numpy()
        vv = sub["v_v"].to_numpy()
        if n > n_points:  # безопасное прореживание до n_points
            idx = np.linspace(0, n - 1, n_points).round().astype(int)
            tt, ii, vv = tt[idx], ii[idx], vv[idx]
            n = n_points
        pad = n_points - n
        tt = np.concatenate([tt, np.full(pad, tt[-1])])
        ii = np.concatenate([ii, np.zeros(pad)])
        vv = np.concatenate([vv, np.full(pad, vv[-1])])
        t_list.append(tt); i_list.append(ii); v_list.append(vv)
    t = torch.tensor(np.array(t_list), dtype=torch.float64)
    i = torch.tensor(np.array(i_list), dtype=torch.float64)
    v = torch.tensor(np.array(v_list), dtype=torch.float64)
    mask = torch.arange(n_points).expand(len(cycle_ids), n_points).clone()
    # маска по дополненным точкам
    for b, cid in enumerate(cycle_ids):
        n_real = min((df["cycle"] == cid).sum(), n_points)
        mask[b, n_real:] = False
    return t, i, v, mask.bool()


# --- идентификация -----------------------------------------------------------

@dataclass
class IdentifiedCell:
    """Результат идентификации элемента."""

    cell_id: str
    c_n: list = field(default_factory=list)
    c_p: list = field(default_factory=list)
    rho: float = 0.0
    q_n_ah: float = 0.0
    j0_mult: tuple = (1.0, 1.0)
    cycles: list = field(default_factory=list)     # номера циклов
    theta_n0: list = field(default_factory=list)
    theta_p0: list = field(default_factory=list)
    r_total_ohm: list = field(default_factory=list)
    q_li_ah: list = field(default_factory=list)
    hyst_mv: list = field(default_factory=list)    # полуразмах гистерезиса
    lam_n: list = field(default_factory=list)      # относит. ёмкость анода
    lam_p: list = field(default_factory=list)      # относит. ёмкость катода
    v_rmse_mv: float = 0.0
    runtime_s: float = 0.0
    ocv_p_scale: float = 1.0                       # аффинная коррекция кривой катода
    ocv_p_shift_v: float = 0.0
    ocv_n_scale: float = 1.0                       # аффинная коррекция кривой анода
    ocv_n_shift_v: float = 0.0
    anchor_name: str = ""
    ocv_p_delta: list = field(default_factory=list)  # поправки кривых, В
    ocv_n_delta: list = field(default_factory=list)
    v_hat: object = None                           # (B,N) тензор предсказанного V
    debug_terms: dict = field(default_factory=dict)  # члены разложения v_hat


def _parse_anchor(rec) -> tuple[torch.Tensor, float, float]:
    """Якорь: новый формат ``{code, scale, shift}`` или старый ``[c...]``."""
    if isinstance(rec, dict):
        return (torch.tensor(rec["code"], dtype=torch.float64),
                float(rec.get("scale", 1.0)), float(rec.get("shift", 0.0)))
    return torch.tensor(rec, dtype=torch.float64), 1.0, 0.0


def _init_c_n(anchors: dict) -> tuple[torch.Tensor, float, float]:
    """Код анода и его аффинная пара: якорь графита Chen2020."""
    for key in ("graphite_Chen2020", "graphite_Ecker2015", "graphite_Mohtat2020"):
        if key in anchors.get("negative", {}):
            return _parse_anchor(anchors["negative"][key])
    return torch.zeros(4, dtype=torch.float64), 1.0, 0.0


def _candidate_c_p(anchors: dict) -> list[tuple[str, torch.Tensor, float, float]]:
    """Кандидаты кода катода: ``(имя, код, scale, shift)`` якорей."""
    out = []
    for key, rec in anchors.get("positive", {}).items():
        code, s, b = _parse_anchor(rec)
        out.append((key, code, s, b))
    return out or [("zero", torch.zeros(4, dtype=torch.float64), 1.0, 0.0)]


def _fit_pseudo_ocv(
    pseudo: pl.DataFrame,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    cp0: torch.Tensor,
    sp0: float,
    bp0: float,
    c_n: torch.Tensor,
    sn0: float,
    bn0: float,
    q0_ah: float,
    steps: int = 700,
    lr: float = 2e-2,
    seed: int = 0,
    with_delta: bool = True,
) -> dict:
    """Подгонка равновесной части по псевдо-OCV формовочных циклов.

    ``V(q) = U_p(θ_p0 − ρ·q/Q_n) − U_n(θ_n0 + q/Q_n)``; оптимизируются
    окна ``(θ_n0, θ_p0)``, ёмкости ``(Q_n, ρ)``, код катода и поправки
    кривых. Возвращает словарь начальных значений для основной
    идентификации.
    """
    q = torch.tensor(pseudo["q_ah"].to_numpy(), dtype=torch.float64)
    v = torch.tensor(pseudo["v"].to_numpy(), dtype=torch.float64)
    q = q - q.min()
    torch.manual_seed(seed)
    G = len(ocv_p.theta_grid)
    tn0 = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    tp0 = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    qn_r = torch.tensor(math.log(math.expm1(max(q0_ah * 1.4, 1e-4))),
                        dtype=torch.float64, requires_grad=True)
    rho_r = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    cp = cp0.clone().requires_grad_(True)
    sp_r = torch.tensor(math.log(max(sp0, 1e-6)), dtype=torch.float64,
                        requires_grad=True)
    bp_v = torch.tensor(bp0, dtype=torch.float64, requires_grad=True)
    sn_r = torch.tensor(math.log(max(sn0, 1e-6)), dtype=torch.float64,
                        requires_grad=True)
    bn_v = torch.tensor(bn0, dtype=torch.float64, requires_grad=True)
    dp = torch.zeros(G, dtype=torch.float64, requires_grad=with_delta)
    dn = torch.zeros(G, dtype=torch.float64, requires_grad=with_delta)
    params = [tn0, tp0, qn_r, rho_r, sp_r, bp_v, sn_r, bn_v]
    if with_delta:
        params += [cp, dp, dn]   # код уточняется только с поправками —
                               # без них форма якоря сравнивается «как есть»
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        th_n = -0.2 + 1.4 * tn0.sigmoid()
        th_p = -0.2 + 1.4 * tp0.sigmoid()
        qn = torch.nn.functional.softplus(qn_r) + 1e-6
        rho = 0.2 + 4.8 * rho_r.sigmoid()
        thn = th_n + q / qn
        thp = th_p - rho * q / qn
        dp_u = dp if with_delta else None
        dn_u = dn if with_delta else None
        u_p = _u_eval(ocv_p, thp, cp, (torch.exp(sp_r), bp_v), dp_u)
        u_n = _u_eval(ocv_n, thn, c_n, (torch.exp(sn_r), bn_v), dn_u)
        pred = u_p - u_n
        loss = ((pred - v) ** 2).mean() \
            + 2e-3 * (dp ** 2).mean() + 2e-3 * (dn ** 2).mean() \
            + 5e-2 * (dp.diff(2) ** 2).mean() + 5e-2 * (dn.diff(2) ** 2).mean()
        if not torch.isfinite(loss):            # защита от расходимости
            return {"rmse_mv": float("inf"), "anchor": "", "theta_n0": 0.15,
                    "theta_p0": 0.9, "q_n_ah": q0_ah / 0.75, "rho": 1.1,
                    "c_p": cp0.detach(), "sp": sp0, "bp": bp0,
                    "sn": sn0, "bn": bn0, "delta_p": torch.zeros(G),
                    "delta_n": torch.zeros(G)}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sch.step()
    with torch.no_grad():
        rmse_t = torch.sqrt(torch.mean((pred - v) ** 2))
        rmse = float(rmse_t) * 1000 if torch.isfinite(rmse_t) else float("inf")
        tn0_v = float(-0.2 + 1.4 * tn0.detach().sigmoid())
        tp0_v = float(-0.2 + 1.4 * tp0.detach().sigmoid())
        qn_v = float(qn.detach())
        rho_v = float(rho.detach())
        sp_v = float(torch.exp(sp_r.detach()))
        sn_v = float(torch.exp(sn_r.detach()))
    return {
        "rmse_mv": rmse,
        "theta_n0": tn0_v, "theta_p0": tp0_v,
        "q_n_ah": qn_v, "rho": rho_v,
        "c_p": cp.detach(), "sp": sp_v,
        "bp": float(bp_v.detach()), "sn": sn_v,
        "bn": float(bn_v.detach()), "delta_p": dp.detach(), "delta_n": dn.detach(),
    }


def identify_cell(
    cell_id: str,
    df: pl.DataFrame,
    cycles: pl.DataFrame,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    anchors: dict,
    formation_cycles: int,
    area_m2: float = 1.54e-4,
    stride: int = 5,
    iters_a: int = 300,
    iters_b: int = 500,
    iters_c: int = 400,
    lr: float = 3e-2,
    v_min: float = 2.5,
    v_max: float = 4.2,
    verbose: bool = False,
    anchor: str | None = None,
    pseudo: pl.DataFrame | None = None,
    pseudo_steps: int = 700,
    max_cycle: int | None = None,
) -> IdentifiedCell:
    """Идентификация параметров одного элемента по его циклам.

    ``df`` — прореженный кадр preprocess; ``cycles`` — поцикловая таблица
    (ёмкости, SOH). ``v_min``/``v_max`` — отсечки основного протокола.
    """
    t_start = time.time()
    res = IdentifiedCell(cell_id=cell_id)
    main_ids = cycles.filter(pl.col("cycle") > formation_cycles)["cycle"].to_list()
    if max_cycle is not None:                     # префикс-режим прогноза
        main_ids = [c for c in main_ids if c <= max_cycle]
    # для коротких префиксов шаг уменьшается, чтобы набрать точек
    stride_eff = max(1, min(stride, len(main_ids) // 20))
    avail = set(df["cycle"].unique().to_list())   # артефактные циклы
    sub_ids = [c for c in main_ids[::stride_eff] if c in avail]
    t, i, v, mask = pack_cycles(df, sub_ids)
    B, N = i.shape

    def inv_softplus(y: float) -> float:
        return math.log(math.expm1(y))

    def rho_of(raw: torch.Tensor) -> torch.Tensor:
        return 0.2 + 4.8 * raw.sigmoid()          # ρ ∈ (0.2, 5)

    def qn_of(raw: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(raw) + 1e-6

    def theta_of(raw: torch.Tensor) -> torch.Tensor:
        # окно стехиометрии расширено до (−0,2; 1,2): выход за [0,1]
        # обслуживается линейной экстраполяцией кривой по краевому наклону
        return -0.2 + 1.4 * raw.sigmoid()

    def lam_of(raw: torch.Tensor) -> torch.Tensor:
        # поцикловый множитель ёмкости электрода (LAM) ∈ (0.4, 1.6)
        return 0.4 + 1.2 * raw.sigmoid()

    # --- инициализация констант ---
    q0 = float(cycles.filter(pl.col("cycle") == main_ids[0])["q_dchg_ah"][0])
    q_n_raw = torch.tensor(inv_softplus(q0 / 0.75), dtype=torch.float64,
                           requires_grad=True)
    rho_raw = torch.tensor(math.log((1.1 - 0.2) / (4.8 - 0.9)),
                           dtype=torch.float64, requires_grad=True)
    # j0 в A/м² — обменный ток обоих электродов; старт ~50 А/м² (быстрая
    # кинетика: остаточная БВ-поляризация при C/2…1C — единицы мВ)
    log_j0n = torch.tensor(math.log(50.0), dtype=torch.float64, requires_grad=True)
    log_j0p = torch.tensor(math.log(50.0), dtype=torch.float64, requires_grad=True)
    c_n, sn0, bn0 = _init_c_n(anchors)

    # --- фаза 0: равновесная подгонка по псевдо-OCV формовочных циклов ---
    # выбор якоря катода и инициализация окон/ёмкостей по квази-
    # равновесной кривой — это отделяет задачу «экспонента» от «кинетики»
    pseudo_fit = None
    if pseudo is not None and pseudo.height > 30:
        cands = _candidate_c_p(anchors)
        if anchor is not None:
            cands = [c for c in cands if c[0] == anchor]
        scored = []
        for name, cp0, s0, b0 in cands:
            # разведка без поправок: сравнение якорей по форме как есть
            f = _fit_pseudo_ocv(pseudo, ocv_n, ocv_p, cp0, s0, b0,
                                c_n, sn0, bn0, q0, steps=pseudo_steps,
                                with_delta=False)
            scored.append((f["rmse_mv"], name, cp0, s0, b0))
        scored.sort(key=lambda x: x[0])
        # инициализация с поправками — для выбранного якоря
        _, best_name, cp0, s0, b0 = scored[0]
        pseudo_fit = _fit_pseudo_ocv(pseudo, ocv_n, ocv_p, cp0, s0, b0,
                                     c_n, sn0, bn0, q0, steps=pseudo_steps,
                                     with_delta=True)
        pseudo_fit["anchor"] = best_name
        if verbose:
            tops = ", ".join(f"{s[1]}:{s[0]:.0f}" for s in scored[:4])
            print(f"  якоря по псевдо-OCV без поправок (мВ): {tops}")
            print(f"  выбран {best_name}; псевдо-OCV RMSE "
                  f"{pseudo_fit['rmse_mv']:.1f} мВ; θ_n0={pseudo_fit['theta_n0']:.3f} "
                  f"θ_p0={pseudo_fit['theta_p0']:.3f} "
                  f"Q_n={pseudo_fit['q_n_ah']*1000:.2f} мА·ч "
                  f"ρ={pseudo_fit['rho']:.3f}")

    def theta_raw(v: float) -> float:
        # обратное к theta_of: raw = logit((θ+0.2)/1.4)
        z = min(max((v + 0.2) / 1.4, 1e-6), 1 - 1e-6)
        return math.log(z / (1 - z))

    # --- поцикловые параметры (в логит-/лог-координатах) ---
    tn0_init = pseudo_fit["theta_n0"] if pseudo_fit else 0.15
    tp0_init = pseudo_fit["theta_p0"] if pseudo_fit else 0.75
    theta_n0 = torch.full((B,), theta_raw(tn0_init), dtype=torch.float64).requires_grad_()
    theta_p0 = torch.full((B,), theta_raw(tp0_init), dtype=torch.float64).requires_grad_()
    r_total = torch.full((B,), math.log(0.03), dtype=torch.float64).requires_grad_()
    # полуразмах гистерезиса ветви заряд/разряд: 80 мВ·tanh (поцикловый)
    hyst_raw = torch.zeros(B, dtype=torch.float64).requires_grad_()
    lam_n_raw = torch.zeros(B, dtype=torch.float64).requires_grad_()
    lam_p_raw = torch.zeros(B, dtype=torch.float64).requires_grad_()
    if pseudo_fit:
        with torch.no_grad():
            q_n_raw.fill_(inv_softplus(pseudo_fit["q_n_ah"]))
            rho_raw.fill_(math.log((pseudo_fit["rho"] - 0.2) / (4.8 - pseudo_fit["rho"])))

    # измеренная ёмкость подмножества циклов — для члена согласованности
    q_meas = torch.tensor(
        [
            float(cycles.filter(pl.col("cycle") == cid)["q_dchg_ah"][0] or 0.0)
            for cid in sub_ids
        ],
        dtype=torch.float64,
    )
    q_meas_n = q_meas / q_meas[0].clamp_min(1e-9)

    # кулонометрический дефицит — независимый от V(t) канал LLI:
    # литий, потерянный за цикл, = q_chg − q_dchg (не возвращается при
    # разряде); кумулятивная сумма по основным циклам ограничивает
    # траекторию инвентаря и частично снимает вырожденность λ↔θ
    _cyc_tab = (
        cycles.filter(pl.col("cycle") > formation_cycles)
        .sort("cycle")
        .select("cycle", "q_chg_ah", "q_dchg_ah", "ce")
    )
    _cy_all = _cyc_tab["cycle"].to_numpy()
    # дефицит = q_chg − q_dchg — верхняя граница LLI: при кинетическом
    # усечении разряда (большой R) он включает невыданную ёмкость,
    # а не потерю лития — такие циклы (CE вне [0.85, 1.05]) исключаем
    _ce = np.nan_to_num(_cyc_tab["ce"].to_numpy().astype(float), nan=0.0)
    _ok_ce = (_ce >= 0.85) & (_ce <= 1.05)
    _def_all = np.where(
        _ok_ce,
        np.clip(np.nan_to_num(
            (_cyc_tab["q_chg_ah"] - _cyc_tab["q_dchg_ah"]).to_numpy()
            .astype(float), nan=0.0), 0.0, None),
        0.0)
    ce_def = torch.tensor(
        [_def_all[(_cy_all > sub_ids[0]) & (_cy_all <= cid)].sum()
         for cid in sub_ids],
        dtype=torch.float64)

    # аффинные пары кривых: инициализация из якорей/фазы 0
    sp0_init = pseudo_fit["sp"] if pseudo_fit else 1.0
    bp0_init = pseudo_fit["bp"] if pseudo_fit else 0.0
    sp_raw = torch.tensor(math.log(max(sp0_init, 1e-6)), dtype=torch.float64,
                          requires_grad=True)                     # ln scale
    bp = torch.tensor(bp0_init, dtype=torch.float64, requires_grad=True)
    sn0_init = pseudo_fit["sn"] if pseudo_fit else sn0
    bn0_init = pseudo_fit["bn"] if pseudo_fit else bn0
    sn_raw = torch.tensor(math.log(max(sn0_init, 1e-6)), dtype=torch.float64,
                          requires_grad=True)
    bn = torch.tensor(bn0_init, dtype=torch.float64, requires_grad=True)

    # аддитивные поправки кривых на сетке θ (в вольтах; из фазы 0 или нули)
    G = len(ocv_p.theta_grid)
    delta_p = (pseudo_fit["delta_p"].clone() if pseudo_fit
               else torch.zeros(G, dtype=torch.float64)).requires_grad_()
    delta_n = (pseudo_fit["delta_n"].clone() if pseudo_fit
               else torch.zeros(G, dtype=torch.float64)).requires_grad_()

    def loss_fn(cp_local, tn0, tp0, rt, qn_r, rho_r, lj0n, lj0p,
                dp=None, dn=None, hy=None, ln=None, lp=None,
                with_cap: bool = True):
        out = simulate_batch(
            t, i, mask, ocv_n, ocv_p,
            theta_of(tn0), theta_of(tp0), qn_of(qn_r), rho_of(rho_r),
            rt.exp(), torch.exp(lj0n), torch.exp(lj0p), area_m2, c_n, cp_local,
            lam_n=lam_of(ln) if ln is not None else None,
            lam_p=lam_of(lp) if lp is not None else None,
            ocv_p_affine=(torch.exp(sp_raw), bp),
            ocv_n_affine=(torch.exp(sn_raw), bn),
            ocv_p_delta=dp, ocv_n_delta=dn,
            hyst_v=0.08 * hy.tanh() if hy is not None else None,
        )
        l_volt = _huber(out["v_hat"], v, mask)
        if with_cap:
            q_pred = theta_of(tn0) * qn_of(qn_r) + theta_of(tp0) * qn_of(qn_r) / rho_of(rho_r)
            q_n_n = q_pred / q_pred[0].clamp_min(1e-9)
            l_cap = torch.mean((q_n_n - q_meas_n) ** 2)
            # CE-член: инвентарь на цикле m ограничен сверху дефицитом
            # кулоновской эффективности — канал, независимый от V(t);
            # лямбда не может «съесть» то, что уже ушло по балансу заряда
            l_ce = torch.mean(
                ((q_pred - (q_pred[0] - ce_def)) / q_meas[0].clamp_min(1e-9)) ** 2)
        else:
            l_cap = l_ce = torch.zeros((), dtype=torch.float64)
        return l_volt + 0.5 * l_cap + 0.5 * l_ce, out

    # --- выбор кода катода ---
    if pseudo_fit:
        # якорь уже выбран и уточнён по псевдо-OCV в фазе 0
        c_p = pseudo_fit["c_p"].clone().requires_grad_(True)
        best = (c_p.detach(), pseudo_fit["rmse_mv"], pseudo_fit["anchor"],
                pseudo_fit["sp"], pseudo_fit["bp"])
    else:
        # запасной путь без псевдо-OCV: короткая динамическая разведка
        best = (None, torch.inf, "none", 1.0, 0.0)
        seen = set()
        for name, cp0, s0, b0 in _candidate_c_p(anchors):
            if anchor is not None and name != anchor:
                continue
            key = tuple(round(x, 3) for x in cp0.tolist())
            if key in seen:
                continue
            seen.add(key)
            with torch.no_grad():
                sp_raw.fill_(math.log(max(s0, 1e-6)))
                bp.fill_(b0)
            cp = cp0.clone().requires_grad_(True)
            opt = torch.optim.Adam([cp, theta_n0, theta_p0, r_total], lr=lr)
            err = None
            for _ in range(120):
                opt.zero_grad()
                err, _ = loss_fn(cp, theta_n0, theta_p0, r_total,
                                 q_n_raw, rho_raw, log_j0n, log_j0p)
                err.backward()
                opt.step()
            if float(err.detach()) < best[1]:
                best = (cp0.detach(), float(err.detach()), name, s0, b0)
        with torch.no_grad():
            sp_raw.fill_(math.log(max(best[3], 1e-6)))
            bp.fill_(best[4])
        c_p = best[0].clone().requires_grad_(True)
    if verbose and not pseudo_fit:
        print(f"  выбран якорь катода {best[2]}; стартовая невязка {best[1]:.4f}")

    # --- фаза A: поцикловые параметры при замороженных константах ---
    opt_a = torch.optim.Adam([theta_n0, theta_p0, r_total], lr=lr)
    for _ in range(iters_a):
        opt_a.zero_grad()
        loss, _ = loss_fn(c_p.detach(), theta_n0, theta_p0, r_total,
                          q_n_raw.detach(), rho_raw.detach(),
                          log_j0n.detach(), log_j0p.detach())
        loss.backward()
        opt_a.step()

    # --- фаза B: совместная подгонка констант и поцикловых параметров ---
    tn0_ref, tp0_ref = theta_n0.detach().clone(), theta_p0.detach().clone()
    params = [theta_n0, theta_p0, r_total, q_n_raw, rho_raw,
              log_j0n, log_j0p, c_p, sp_raw, bp, sn_raw, bn,
              delta_p, delta_n, hyst_raw, lam_n_raw, lam_p_raw]
    opt_b = torch.optim.Adam(params, lr=lr * 0.3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=iters_b)
    out = None
    for it in range(iters_b):
        opt_b.zero_grad()
        loss, out = loss_fn(c_p, theta_n0, theta_p0, r_total,
                            q_n_raw, rho_raw, log_j0n, log_j0p,
                            dp=delta_p, dn=delta_n, hy=hyst_raw,
                            ln=lam_n_raw, lp=lam_p_raw)
        # мягкий априор: удержание окон вблизи решения фазы A; гребневая
        # регуляризация и гладкость поправок кривых (вторая разность)
        loss = loss + 1e-4 * ((theta_n0 - tn0_ref) ** 2).mean() \
                  + 1e-4 * ((theta_p0 - tp0_ref) ** 2).mean() \
                  + 2e-3 * (delta_p ** 2).mean() \
                  + 2e-3 * (delta_n ** 2).mean() \
                  + 5e-2 * (delta_p.diff(2) ** 2).mean() \
                  + 5e-2 * (delta_n.diff(2) ** 2).mean()
        loss.backward()
        opt_b.step()
        sched.step()
        if verbose and it % 100 == 0:
            print(f"    B{it}: huber {float(loss.detach()):.5f}")

    # --- фаза C: поцикловое доукрупнение при замороженных константах ---
    opt_c = torch.optim.Adam(
        [theta_n0, theta_p0, r_total, hyst_raw, lam_n_raw, lam_p_raw],
        lr=lr * 0.2)
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=iters_c)
    for it in range(iters_c):
        opt_c.zero_grad()
        loss, out = loss_fn(c_p.detach(), theta_n0, theta_p0, r_total,
                            q_n_raw.detach(), rho_raw.detach(),
                            log_j0n.detach(), log_j0p.detach(),
                            dp=delta_p.detach(), dn=delta_n.detach(),
                            hy=hyst_raw, ln=lam_n_raw, lp=lam_p_raw)
        loss.backward()
        opt_c.step()
        sched_c.step()
    if verbose:
        print(f"    C-финал: huber {float(loss.detach()):.5f}")

    with torch.no_grad():
        tn = theta_of(theta_n0).detach()
        tp = theta_of(theta_p0).detach()
        rt = r_total.exp().detach()
        q_n_v = float(qn_of(q_n_raw.detach()))
        rho_v = float(rho_of(rho_raw.detach()))
        q_p_v = q_n_v / rho_v
        q_li = tn * q_n_v + tp * q_p_v   # А·ч-эквивалент запаса лития
        res.c_n = [float(x) for x in c_n.tolist()]
        res.c_p = [float(x) for x in c_p.detach().tolist()]
        res.rho = rho_v
        res.q_n_ah = q_n_v
        res.j0_mult = (float(torch.exp(log_j0n)), float(torch.exp(log_j0p)))
        res.cycles = [int(x) for x in sub_ids]
        res.theta_n0 = [float(x) for x in tn.tolist()]
        res.theta_p0 = [float(x) for x in tp.tolist()]
        res.r_total_ohm = [float(x) for x in rt.tolist()]
        res.q_li_ah = [float(x) for x in q_li.tolist()]
        res.hyst_mv = [float(0.08 * torch.tanh(x) * 1000)
                       for x in hyst_raw.detach()]
        res.lam_n = [float(x) for x in lam_of(lam_n_raw).detach().tolist()]
        res.lam_p = [float(x) for x in lam_of(lam_p_raw).detach().tolist()]
        res.ocv_p_scale = float(torch.exp(sp_raw.detach()))
        res.ocv_p_shift_v = float(bp.detach())
        res.ocv_n_scale = float(torch.exp(sn_raw.detach()))
        res.ocv_n_shift_v = float(bn.detach())
        res.anchor_name = best[2]
        res.ocv_p_delta = [float(x) for x in delta_p.detach().tolist()]
        res.ocv_n_delta = [float(x) for x in delta_n.detach().tolist()]
        res.v_hat = out["v_hat"].detach()
        res.debug_terms = {k: out[k].detach() for k in
                           ("eta_p", "eta_n", "v_diff", "u_p", "u_n",
                            "theta_n", "theta_p")}
        res.v_rmse_mv = float(torch.sqrt(torch.mean(
            ((out["v_hat"] - v) ** 2)[mask]))) * 1000.0
    res.runtime_s = time.time() - t_start
    return res


def refine_per_cycle(
    res: IdentifiedCell,
    df: pl.DataFrame,
    cycles: pl.DataFrame,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    area_m2: float = 1.54e-4,
    iters: int = 250,
    lr: float = 1e-2,
    w_cont: float = 1e-3,
    verbose: bool = False,
) -> IdentifiedCell:
    """Независимое поцикловое уточнение после совместной идентификации.

    Оператор ``i(t) → v(t)`` каждого цикла инвертируется почти независимо:
    на цикл приходятся свои ``θ_n0, θ_p0, R, j0_n, j0_p, λ_n, λ_p, hyst``
    (тёплый старт — совместное решение), общими остаются только
    равновесные кривые ``(c, affine, δ-поправки)`` и опорные ёмкости
    ``Q_n, ρ``. Мягкий штраф ``w_cont·‖ψ_k − ψ_{k−1}‖²`` удерживает
    непрерывность траектории там, где один цикл параметры не фиксирует.

    Результат — функции параметров ψ(k), которые далее используются как
    наблюдаемые для модели деградации (этап 2).
    """
    t, i, v, mask = pack_cycles(df, res.cycles)
    B, N = i.shape

    def theta_of(raw):
        return -0.2 + 1.4 * raw.sigmoid()

    def theta_raw(val):
        z = min(max((val + 0.2) / 1.4, 1e-6), 1 - 1e-6)
        return math.log(z / (1 - z))

    def lam_of(raw):
        return 0.4 + 1.2 * raw.sigmoid()

    def lam_raw(val):
        z = min(max((val - 0.4) / 1.2, 1e-6), 1 - 1e-6)
        return math.log(z / (1 - z))

    def tr(v, fn):
        return torch.tensor([fn(x) for x in v], dtype=torch.float64)

    tn0 = tr(res.theta_n0, theta_raw).requires_grad_()
    tp0 = tr(res.theta_p0, theta_raw).requires_grad_()
    rt = torch.log(torch.tensor(res.r_total_ohm, dtype=torch.float64)).requires_grad_()
    lj0n = torch.full((B,), math.log(max(res.j0_mult[0], 1e-6)),
                      dtype=torch.float64).requires_grad_()
    lj0p = torch.full((B,), math.log(max(res.j0_mult[1], 1e-6)),
                      dtype=torch.float64).requires_grad_()
    ln = tr(res.lam_n or [1.0] * B, lam_raw).requires_grad_()
    lp = tr(res.lam_p or [1.0] * B, lam_raw).requires_grad_()
    hy = tr([h / 80.0 for h in (res.hyst_mv or [0.0] * B)],
            lambda z: 0.5 * math.log((1 + min(max(z, -0.99), 0.99))
                                     / (1 - min(max(z, -0.99), 0.99)))).requires_grad_()

    q_n = torch.tensor(res.q_n_ah, dtype=torch.float64)
    rho = torch.tensor(res.rho, dtype=torch.float64)
    c_n = torch.tensor(res.c_n, dtype=torch.float64)
    c_p = torch.tensor(res.c_p, dtype=torch.float64)
    aff_p = (torch.tensor(res.ocv_p_scale, dtype=torch.float64),
             torch.tensor(res.ocv_p_shift_v, dtype=torch.float64))
    aff_n = (torch.tensor(res.ocv_n_scale, dtype=torch.float64),
             torch.tensor(res.ocv_n_shift_v, dtype=torch.float64))
    dp = torch.tensor(res.ocv_p_delta or [0.0] * len(ocv_p.theta_grid),
                      dtype=torch.float64)
    dn = torch.tensor(res.ocv_n_delta or [0.0] * len(ocv_n.theta_grid),
                      dtype=torch.float64)
    q_meas = torch.tensor(
        [float(cycles.filter(pl.col("cycle") == cid)["q_dchg_ah"][0] or 0.0)
         for cid in res.cycles], dtype=torch.float64)
    q_meas_n = q_meas / q_meas[0].clamp_min(1e-9)

    params = [tn0, tp0, rt, lj0n, lj0p, ln, lp, hy]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    out = None
    for it in range(iters):
        opt.zero_grad()
        out = simulate_batch(
            t, i, mask, ocv_n, ocv_p,
            theta_of(tn0), theta_of(tp0), q_n, rho, rt.exp(),
            lj0n.exp().unsqueeze(1), lj0p.exp().unsqueeze(1),
            area_m2, c_n, c_p,
            lam_n=lam_of(ln), lam_p=lam_of(lp),
            ocv_p_affine=aff_p, ocv_n_affine=aff_n,
            ocv_p_delta=dp, ocv_n_delta=dn,
            hyst_v=0.08 * hy.tanh())
        l_volt = _huber(out["v_hat"], v, mask)
        q_pred = theta_of(tn0) * q_n * lam_of(ln) \
            + theta_of(tp0) * q_n * lam_of(lp) / rho
        l_cap = torch.mean((q_pred / q_pred[0].clamp_min(1e-9) - q_meas_n) ** 2)
        # непрерывность траекторий в логит-/лог-координатах
        l_cont = sum(((p[1:] - p[:-1]) ** 2).mean()
                     for p in (tn0, tp0, rt, ln, lp))
        loss = l_volt + 0.5 * l_cap + w_cont * l_cont
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % 50 == 0:
            print(f"    D{it}: volt {float(l_volt.detach()):.5f} "
                  f"cont {float(l_cont.detach()):.5f}")

    with torch.no_grad():
        tn_v, tp_v = theta_of(tn0), theta_of(tp0)
        lam_n_v, lam_p_v = lam_of(ln), lam_of(lp)
        res.theta_n0 = [float(x) for x in tn_v]
        res.theta_p0 = [float(x) for x in tp_v]
        res.r_total_ohm = [float(x) for x in rt.exp()]
        res.lam_n = [float(x) for x in lam_n_v]
        res.lam_p = [float(x) for x in lam_p_v]
        res.hyst_mv = [float(0.08 * torch.tanh(x) * 1000) for x in hy]
        res.j0_mult = (float(lj0n.exp().mean()), float(lj0p.exp().mean()))
        res.j0n_per_cycle = [float(x) for x in lj0n.exp()]
        res.j0p_per_cycle = [float(x) for x in lj0p.exp()]
        res.q_li_ah = [float(x) for x in
                       (tn_v * q_n * lam_n_v + tp_v * q_n * lam_p_v / rho)]
        res.v_hat = out["v_hat"].detach()
        res.debug_terms = {k: out[k].detach() for k in
                           ("eta_p", "eta_n", "v_diff", "u_p", "u_n",
                            "theta_n", "theta_p")}
        res.v_rmse_mv = float(torch.sqrt(torch.mean(
            ((out["v_hat"] - v) ** 2)[mask]))) * 1000.0
    return res


def _huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
           delta: float = 0.02) -> torch.Tensor:
    """Функция Хьюбера по точкам маски; delta=20 мВ."""
    d = (pred - target)[mask]
    hub = torch.where(d.abs() <= delta, 0.5 * d ** 2, delta * (d.abs() - 0.5 * delta))
    return hub.mean()


def load_pretrained_ocv(ckpt_dir: str | Path, latent_dim: int = 4) -> tuple[MonotoneOCV, MonotoneOCV]:
    """Загружает веса предобученных OCV-сетей этапа 0."""
    ckpt_dir = Path(ckpt_dir)
    def load(name):
        blob = torch.load(ckpt_dir / name, map_location="cpu", weights_only=False)
        net = MonotoneOCV(latent_dim=blob.get("latent_dim", latent_dim), hidden=96, n_layers=3)
        net.load_state_dict(blob["state_dict"])
        return net
    return load("ocv_n.pt"), load("ocv_p.pt")


def load_anchors(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
