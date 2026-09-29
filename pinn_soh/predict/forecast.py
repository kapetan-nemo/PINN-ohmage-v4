"""Рекурсивный прогноз деградации и напряжения (режим оператора).

Сценарий: по первым ``K`` циклам выполнена идентификация состояния
(``IdentifiedCell``) и калибровка кинетики ``z``. Далее состояние
рекурсивно разворачивается на произвольное число циклов:

1. баланс лития + отсечки напряжения → стехиометрическое окно цикла;
2. замкнутая симуляция цикла по заданному шаблону тока → ``φ_n(t)``,
   ``v(t)``;
3. интегралы паразитных токов (SEI, осаждение) → приращения ``ΔL_N``,
   ``Δδ_SEI``;
4. обновление состояния, переход к следующему циклу.

Чтобы ускорить длинные горизонты, симуляция выполняется на каждом
``stride``-м цикле, а приращение умножается на шаг (медленная динамика).
Будущий токовый профиль задаётся шаблоном ``(t, i)`` — по умолчанию
последний наблюдённый цикл (повторяющийся основной протокол).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pinn_soh.physics.cell import F_CONST, R_CONST, T_REF
from pinn_soh.physics.degradation import DegradationConsts
from pinn_soh.physics.ocv import MonotoneOCV, torchinterp1
from pinn_soh.train.stage1_extract_state import (
    IdentifiedCell, _u_eval, pack_cycles, simulate_batch,
)


@dataclass
class ForecastResult:
    """Результат рекурсивного прогноза."""

    cycles: np.ndarray            # номера спрогнозированных циклов
    soh: np.ndarray               # предсказанная SOH (норм. к q_ref)
    q_li_ah: np.ndarray           # запас лития
    r_total_ohm: np.ndarray       # полное сопротивление
    delta_sei_m: np.ndarray       # толщина SEI
    q_dch_ah: np.ndarray          # предсказанная разрядная ёмкость
    theta_n0: np.ndarray          # окно анода в начале цикла
    theta_p0: np.ndarray
    lam_n: np.ndarray | None = None   # множители ёмкости электродов
    lam_p: np.ndarray | None = None
    q_dch_raw: np.ndarray | None = None  # ёмкость до проекции на невозрастание
    q_win: np.ndarray | None = None      # термодинамическая ёмкость окна
    q_sim: np.ndarray | None = None      # кинетическая ёмкость (симуляция)
    v_hat: np.ndarray | None = None   # (B,N) предсказанные кривые (опц.)
    gains: np.ndarray | None = None   # поэлементные поправки переходной модели
    floors: np.ndarray | None = None  # персистентность каналов с префикса


def _u_full(net: MonotoneOCV, th: torch.Tensor, code: torch.Tensor,
            affine: tuple[torch.Tensor, torch.Tensor],
            delta: torch.Tensor | None) -> torch.Tensor:
    """U(θ) с аффином и сеточной поправкой — как в идентификации."""
    return _u_eval(net, th.reshape(-1), code, affine, delta).reshape(th.shape)


def resolve_windows_full(
    ocv_n: MonotoneOCV, ocv_p: MonotoneOCV,
    c_n: torch.Tensor, c_p: torch.Tensor,
    aff_n, aff_p, d_n, d_p,
    q_n_ah: float, rho: float, q_li_ah: float,
    lam_n: float, lam_p: float,
    v_min: float, v_max: float, n_grid: int = 4000,
) -> dict:
    """Стехиометрические окна при текущем состоянии.

    Баланс: ``θ_n·Q_n·λ_n + θ_p·Q_p·λ_p = q_li``; равновесное напряжение
    элемента ``V(θ_n)`` монотонно по θ_n — отсечки решаются бисекцией.
    """
    q_n, q_p = q_n_ah * lam_n, q_n_ah / rho * lam_p
    # сетка в том же допустимом диапазоне, что у идентификации
    # (theta_of: (−0.2, 1.2)) — иначе элементы с идентифицированным
    # θ>1 молча перепроецируются в [0,1] на стыке (дисконтинуитет
    # состояния); за [0,1] работает линейная экстраполяция _u_eval
    th = torch.linspace(-0.15, 1.15, n_grid, dtype=torch.float64)
    theta_p = (q_li_ah - th * q_n) / q_p
    valid = (theta_p > -0.15) & (theta_p < 1.15)
    if valid.sum() < 2:
        return {"theta_n0": None, "degenerate": True}
    th_v, tp_v = th[valid], theta_p[valid]
    u_p = _u_full(ocv_p, tp_v, c_p, aff_p, d_p)
    u_n = _u_full(ocv_n, th_v, c_n, aff_n, d_n)
    v_cell = (u_p - u_n).detach()
    # OCV элемента физически монотонна: дрожание суррогата на плато
    # создаёт ложные пересечения отсечек и скачки окна — убираем
    v_cell = torch.cummax(v_cell, dim=0).values

    def solve(v_t: float) -> float:
        if v_t <= float(v_cell[0]):
            return float(th_v[0])
        if v_t >= float(v_cell[-1]):
            return float(th_v[-1])
        idx = int(torch.searchsorted(v_cell, v_t))
        x0, x1 = float(th_v[idx - 1]), float(th_v[idx])
        y0, y1 = float(v_cell[idx - 1]), float(v_cell[idx])
        w = (v_t - y0) / max(y1 - y0, 1e-12)
        return x0 + w * (x1 - x0)

    tn_lo, tn_hi = solve(v_min), solve(v_max)
    # ёмкость окна — мягкий индикатор σ(v_min<v<v_max): вклад θ
    # затухает плавно на краях отсечек, убирает дискретный перескок
    # solve() у плато OCV и при пересечении границы валидности
    eps_w = 0.015
    vv = v_cell.numpy()
    w = (1.0 / (1.0 + np.exp(-(vv - v_min) / eps_w))) \
        * (1.0 / (1.0 + np.exp(-(v_max - vv) / eps_w)))
    q_win_soft = float(np.trapezoid(w, th_v.numpy())) * q_n
    # разрядная ёмкость ограничена и шириной стехиометрического окна,
    # и запасом лития — нельзя перенести больше Li, чем есть в элементе
    return {
        "theta_n0": tn_lo,
        "theta_p0": float((q_li_ah - tn_lo * q_n) / q_p),
        "theta_n_chg": tn_hi,
        "theta_p_chg": float((q_li_ah - tn_hi * q_n) / q_p),
        "q_dch_ah": min(q_win_soft, q_li_ah),
        "degenerate": False,
    }


def fit_z_prefix(
    res: IdentifiedCell,
    df,
    consts: DegradationConsts,
    iters: int = 200,
    lr: float = 0.05,
    z0: torch.Tensor | None = None,
    area_m2: float | None = None,
) -> torch.Tensor:
    """Калибровка кинетики z по траектории q_li префикса (grey-box).

    Прямая оптимизация 5 множителей на идентифицированных циклах —
    без кодирующей сети. Используется для оценки прогноза до обучения
    encoder и как поэлементная нижняя граница качества.
    """
    from pinn_soh.train.stage2_degradation import CellData, rollout_state

    t, i, _, _ = pack_cycles(df, res.cycles)
    dt = res.debug_terms
    phi_n = (dt["u_n"] - dt["eta_n"])
    cell = CellData(
        cell_id=res.cell_id, cycles=np.asarray(res.cycles, float),
        q_li=np.asarray(res.q_li_ah, float),
        r_total=np.asarray(res.r_total_ohm, float),
        t_s=t, i_a=i, phi_n=phi_n, q_dch_meas=None,
        feats=torch.zeros(1, dtype=torch.float64),
    )
    z0_ = z0.detach() if z0 is not None else None
    z = (z0_.clone() if z0_ is not None
         else torch.zeros(5, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    q_true = torch.tensor(cell.q_li, dtype=torch.float64)
    r_true = torch.tensor(cell.r_total, dtype=torch.float64)
    for _ in range(iters):
        opt.zero_grad()
        pred = rollout_state(
            cell, z, consts,
            area_m2=getattr(res, "area_m2", None) or area_m2 or 1.54e-4)
        # на коротком префиксе траектория R — ключевое ограничение
        # на ρ_SEI/k_SEI (растёт сильнее, чем падает q_li)
        # априор к z0 (encoder): короткий префикс плохо ограничивает
        # кинетику — остаёмся в окрестности популяционной оценки
        prior = ((z - z0_) ** 2).mean() if z0_ is not None \
            else (z ** 2).mean()
        loss = (((pred["q_li"] - q_true) / q_true[0]) ** 2).mean() \
            + 0.5 * (((pred["r_total"] - r_true) / r_true[0]) ** 2).mean() \
            + 0.05 * prior
        loss.backward()
        torch.nn.utils.clip_grad_value_(z, 10.0)
        opt.step()
    return z.detach()


def forecast(
    res: IdentifiedCell,
    df,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    consts: DegradationConsts,
    z: torch.Tensor,
    cycle_end: int,
    v_min: float,
    v_max: float,
    stride: int = 5,
    area_m2: float = 1.54e-4,
    kinetic_cap: bool = False,
    r_boost: float = 1.0,
    r_film: bool = False,
    r_qli_gamma: float = 0.0,
    ocv_clamp: bool = False,
    i_template: tuple[np.ndarray, np.ndarray] | None = None,
    q_ref_ah: float | None = None,
    save_curves: bool = False,
    rate_match: bool = False,
    trans: torch.nn.Module | None = None,
    feats: torch.Tensor | None = None,
    q_li_start: float | None = None,
    r_fate: tuple[float, float, float] | None = None,
    gains=None,
    lam_snr: bool = True,
) -> ForecastResult:
    """Рекурсивный прогноз от последнего идентифицированного цикла.

    ``i_template`` — (t_rel, i) шаблон будущего цикла; по умолчанию
    последний идентифицированный цикл. ``q_ref_ah`` — опорная ёмкость
    нормировки SOH (по умолчанию — предсказанная ёмкость первого
    прогнозного цикла). ``q_li_start`` — переопределение запаса лития на
    границе (ось неопределённости идентификации: оценка на префиксе
    расходится с полножизненной на ±30%). ``r_fate`` — сценарий
    логистического взрыва сопротивления ``(r_max_ohm, k_mid, w)``:
    R-пол ``r_max·σ((k−k_mid)/w)`` ограничивает сопротивление снизу
    (судьба взрыва не видна на префиксе — см. журнал).
    ``gains`` — явное переопределение префиксной калибровки каналов
    (ветка ансамбля «LAM не персистентен»); ``lam_snr=False`` отключает
    SNR-взвешивание λ-каналов (ветка «LAM персистентен»).
    """
    # шаблон цикла: последний идентифицированный с ТИПИЧНОЙ
    # длительностью — последний цикл префикса может быть усечённым/
    # атипичным, тогда шаблон искажает весь прогнозный профиль
    if i_template is None:
        import polars as pl
        cand = res.cycles[-5:]
        durs = []
        for c_ in cand:
            sub_ = df.filter(pl.col("cycle") == c_)
            durs.append(float(sub_["t_s"].max() - sub_["t_s"].min())
                        if sub_.height else 0.0)
        med_dur = np.median(durs) if durs else 0.0
        cyc_t = next(
            (c_ for c_, d_ in zip(reversed(cand), reversed(durs))
             if d_ >= 0.7 * med_dur), res.cycles[-1])
        sub = df.filter(pl.col("cycle") == cyc_t).sort("t_s")
        tt = sub["t_s"].to_numpy()
        i_template = (tt - tt[0], sub["i_a"].to_numpy())
    tt_rel, ii = i_template
    if kinetic_cap:
        # разрядная часть шаблона конечна: прогнозная ёмкость не
        # должна упираться в её длительность — продлеваем разряд
        # тем же током с запасом ×2 от шаблонного заряда
        dch_idx = np.where(ii < -1e-7)[0]
        if len(dch_idx) > 3:
            i_dch = float(np.median(-ii[dch_idx]))
            dt_med = float(np.median(np.diff(tt_rel)))
            q_cap = float(np.sum(
                -ii[dch_idx]
                * np.gradient(tt_rel[dch_idx]))) / 3600.0
            if i_dch > 0 and dt_med > 0 and q_cap > 0:
                ext_s = 2.0 * q_cap / i_dch * 3600.0
                n_ext = int(ext_s / dt_med) + 1
                cut = dch_idx[-1] + 1
                t_ext = tt_rel[cut - 1] + dt_med * np.arange(1, n_ext + 1)
                shift = t_ext[-1] - tt_rel[cut - 1]
                tt_rel = np.concatenate(
                    [tt_rel[:cut], t_ext, tt_rel[cut:] + shift])
                ii = np.concatenate(
                    [ii[:cut], np.full(n_ext, -i_dch), ii[cut:]])
    t = torch.tensor(tt_rel[None, :], dtype=torch.float64)
    i = torch.tensor(ii[None, :], dtype=torch.float64)
    mask = torch.ones(1, len(ii), dtype=torch.bool)

    c_n = torch.tensor(res.c_n, dtype=torch.float64)
    c_p = torch.tensor(res.c_p, dtype=torch.float64)
    aff_p = (torch.tensor(res.ocv_p_scale, dtype=torch.float64),
             torch.tensor(res.ocv_p_shift_v, dtype=torch.float64))
    aff_n = (torch.tensor(res.ocv_n_scale, dtype=torch.float64),
             torch.tensor(res.ocv_n_shift_v, dtype=torch.float64))
    G = len(ocv_p.theta_grid)
    dp = torch.tensor(res.ocv_p_delta or [0.0] * G, dtype=torch.float64)
    dn = torch.tensor(res.ocv_n_delta or [0.0] * G, dtype=torch.float64)
    j0n = float(np.mean(getattr(res, "j0n_per_cycle", None)
                          or res.j0_mult[0]))
    j0p = float(np.mean(getattr(res, "j0p_per_cycle", None)
                          or res.j0_mult[1]))
    lam_n0 = float(np.mean((res.lam_n or [1.0])[-3:]))
    lam_p0 = float(np.mean((res.lam_p or [1.0])[-3:]))

    def lam_slope(lam: list[float], rate_floor: float) -> float:
        """Линейный тренд относительной ёмкости по последним циклам
        (дрейф LAM), в единицах на цикл; экстраполяция ограничена.
        Наклон слабее шума идентификации (|Δλ|/цикл по популяции)
        трактуется как нулевой — иначе шумовой тренд префикса
        экстраполируется в ложный спад."""
        if len(lam) < 4:
            return 0.0
        m = min(12, len(lam))
        x = np.asarray(res.cycles[-m:], float)
        y = np.asarray(lam[-m:], float)
        s = float(np.polyfit(x - x[0], y, 1)[0])
        # ёмкость электрода физически не растёт: наклон неположителен
        return min(s, 0.0) if abs(s) > rate_floor else 0.0

    # пороги — медианный шум идентификации λ на цикл (по популяции)
    dlam_n = lam_slope(res.lam_n or [1.0], 1.5e-3)
    dlam_p = lam_slope(res.lam_p or [1.0], 6.7e-4)
    qli0_, r0_ = res.q_li_ah[0], res.r_total_ohm[0]
    lam_n, lam_p = lam_n0, lam_p0
    r_state = res.r_total_ohm[-1]
    # инерция переходной модели: последнее наблюдаемое приращение q_li
    if len(res.q_li_ah) >= 2:
        gp = max(res.cycles[-1] - res.cycles[-2], 1)
        dq_prev = max(res.q_li_ah[-2] - res.q_li_ah[-1], 0.0) \
            / gp / qli0_
    else:
        dq_prev = 0.0

    def lam_at(lam0: float, dlam: float, k: int) -> float:
        # границы те же, что у lam_of в идентификации: (0.4, 1.6)
        return float(np.clip(lam0 + dlam * (k - cyc0), 0.4, 1.6))

    # кинетические константы с множителями z
    def _prefix_gains(lam_snr: bool = True):
        """Поэлементные поправки к выходам переходной модели.

        На идентифицированных циклах префикса сравнивает предсказанные
        моделью скорости приращений с наблюдаемыми; медианное отношение
        — множитель канала (ограничен [0.3, 4]). Делает популяционную
        модель адаптивной к конкретному элементу.
        """
        if len(res.cycles) < 4 or trans is None or feats is None:
            return np.ones(4)
        try:
            tt_, ii_, _, _ = pack_cycles(df, res.cycles)
            dt = res.debug_terms
            phi = (dt["u_n"] - dt["eta_n"]).double()
            w_ = torch.zeros(phi.shape, dtype=torch.float64)
            dts_ = tt_[:, 1:] - tt_[:, :-1]
            w_[:, 0], w_[:, -1] = 0.5 * dts_[:, 0], 0.5 * dts_[:, -1]
            w_[:, 1:-1] = 0.5 * (dts_[:, :-1] + dts_[:, 1:])
            expo_ = torch.exp(
                a_sei * (phi - consts.u_sei_v)).clamp(max=1e6)
            expp_ = torch.exp((a_pl * phi).clamp(max=50.0))
            expm_ = torch.exp((-a_pl * phi).clamp(max=50.0))
            j_pl_ = torch.where(phi < 0.0, j0_pl * (expp_ - expm_),
                                torch.zeros_like(phi))
            jpl = (j_pl_.abs() * w_).sum(dim=1)
            lam_np = np.asarray(res.lam_n or [1.0] * len(res.cycles))
            lam_pp = np.asarray(res.lam_p or [1.0] * len(res.cycles))
            q_arr = np.asarray(res.q_li_ah)
            r_arr = np.asarray(res.r_total_ohm)
            cyc = np.asarray(res.cycles, float)
            d_ = consts.delta_sei0_m
            g0 = max(cyc[1] - cyc[0], 1.0)
            dq_p = max(q_arr[0] - q_arr[1], 0.0) / g0 / qli0_
            ratios = np.full((len(cyc) - 1, 4), np.nan)
            for m in range(len(cyc) - 1):
                gap = cyc[m + 1] - cyc[m]
                denom = d_ / d_solv + expo_[m] / k_sei
                j_sei_ = (-F_CONST * consts.c_solv_mol_m3 / denom).abs()
                int_sei_ = float((j_sei_ * w_[m]).sum()) * area_eff / 3600.0
                int_pl_ = float(jpl[m]) * area_eff / 3600.0
                dqm = int_sei_ + consts.beta_dead * int_pl_
                d_ += int_sei_ * 3600.0 * consts.v_sei_m3_mol \
                    / (F_CONST * area_eff)
                dyn_ = torch.tensor(
                    [[q_arr[m] / qli0_, lam_np[m], lam_pp[m],
                      math.log10(max(r_arr[m] / r0_, 1e-6)),
                      dqm / qli0_, math.log10(cyc[m] + 1.0) / 4.0,
                      gap / 20.0, dq_p]], dtype=torch.float64)
                with torch.no_grad():
                    o_ = trans(dyn_, feats.reshape(1, -1))[0]
                dq_p = float(o_[0]) * dqm / qli0_ + float(o_[1])
                pred_q = dq_p * gap
                obs_q = (q_arr[m] - q_arr[m + 1]) / qli0_
                if pred_q > 1e-9 and obs_q > 0:
                    ratios[m, 0] = obs_q / pred_q
                if float(o_[2]) > 1e-9:
                    ratios[m, 1] = (lam_np[m] - lam_np[m + 1]) / gap \
                        / float(o_[2])
                if float(o_[3]) > 1e-9:
                    ratios[m, 2] = (lam_pp[m] - lam_pp[m + 1]) / gap \
                        / float(o_[3])
                obs_r = math.log10(max(r_arr[m + 1] / r_arr[m], 1e-6))
                if float(o_[4]) > 1e-9 and obs_r > 0:
                    ratios[m, 3] = obs_r / gap / float(o_[4])
            # калибровка по ПОЗДНЕЙ половине префикса: ранняя осадка
            # после формовки не должна определять будущий темп;
            # медиана поздних отношений отражает текущий режим
            n_step = len(cyc) - 1
            lo = n_step // 2 if n_step >= 8 else 0
            g_ = np.nanmedian(
                np.where(ratios[lo:] > 0, ratios[lo:], np.nan), axis=0)
            # SNR-взвешивание λ-каналов: наблюдаемый темп спада порядка
            # шума идентификации (|Δλ_n|~1.5e-3, |Δλ_p|~6.7e-4/цикл —
            # значимые пороги как у _prefix_persistence) делает
            # отношение наблюд./модель шумным; вместо жёсткого сброса —
            # плавное стягивание к 1: w = rate/(rate+floor)
            lam_floor = (1.5e-3, 6.7e-4)  # медиана шума идентификации λ
            for ch, arr in ((1, lam_np), (2, lam_pp)):
                obs_rate = np.nanmedian(
                    (arr[lo:-1] - arr[lo + 1:]) / np.diff(cyc[lo:])) \
                    if len(cyc) - lo >= 2 else np.nan
                if lam_snr and np.isfinite(obs_rate):
                    w_snr = np.clip(obs_rate / (obs_rate + lam_floor[ch - 1]),
                                    0.0, 1.0)
                    g_[ch] = 1.0 + (g_[ch] - 1.0) * w_snr
            g_ = np.clip(np.where(np.isfinite(g_), g_, 1.0), 0.1, 4.0)
            # стягивание к 1 при малом числе шагов префикса:
            # медиана по 3–4 отношениям шумна, полная сила с ~12
            w_shrink = min(1.0, n_step / 12.0)
            return 1.0 + (g_ - 1.0) * w_shrink
        except Exception as e:
            # сбой калибровки не должен молча отключать каналы —
            # нейтральные gains скрывают реальные ошибки пайплайна
            import warnings
            warnings.warn(f"_prefix_gains: сбой, gains=1 ({e})")
            return np.ones(4)

    def _prefix_persistence() -> np.ndarray:
        """Поэлементная персистентность каналов p ∈ [0, 1.2] (H9).

        Глобальное затухание k^β верно «в среднем», но не различает
        раннюю осадку и персистентную деградацию. Наблюдаемый критерий:
        отношение темпа убыли во второй половине префикса к первой —
        у здоровых каналы останавливаются (p→0, полное затухание),
        у умирающих темп персистентен или растёт (p→1+, β_eff→0/ускор.).
        """
        if len(res.cycles) < 6:
            return np.full(4, 0.5)
        cyc = np.asarray(res.cycles, float)
        mid = len(cyc) // 2
        if mid < 2 or len(cyc) - mid < 2:
            return np.full(4, 0.5)
        lam_np = np.asarray(res.lam_n or [1.0] * len(cyc), float)
        lam_pp = np.asarray(res.lam_p or [1.0] * len(cyc), float)
        series = [np.asarray(res.q_li_ah, float), lam_np, lam_pp]
        # темп УБЫЛИ: положителен при деградации — для убывающих
        # каналов (q_li, λ) наклон берётся со знаком минус
        rates = [((s[0] - s[mid]) / max(cyc[mid] - cyc[0], 1.0),
                  (s[mid] - s[-1]) / max(cyc[-1] - cyc[mid], 1.0))
                 for s in series]
        r = np.asarray(res.r_total_ohm, float)
        eps_r = np.maximum(r, 1e-12)
        rates.append((math.log10(eps_r[mid] / eps_r[0])
                      / max(cyc[mid] - cyc[0], 1.0),
                      math.log10(eps_r[-1] / eps_r[mid])
                      / max(cyc[-1] - cyc[mid], 1.0)))
        # пороги значимости темпа: ниже шума идентификации темп —
        # артефакт оценки, а не судьба. Популяционный шум |Δ|/цикл:
        # q_li ~ 1.4e-6 (p90 4e-6), λ_n ~ 1.5e-3 (p90 6.4e-3),
        # λ_p ~ 6.7e-4 (p90 2.8e-3). Реальные темпы λ (1–3e-3/цикл)
        # сравнимы с шумом — персистентность λ по короткому префиксу
        # неопределима, пороги взяты ≈p85 шума иначе шумовой «поздний
        # старт» давал ложное ускорение LAM и гибель здоровых элементов.
        rate_floor = (1e-6, 4e-3, 2e-3, 2e-4)   # q_li А·ч/ц, λ_n/ц, λ_p/ц, log10R/ц
        pers = np.full(4, 0.5)
        for i, (e_, l_) in enumerate(rates):
            if e_ > rate_floor[i]:
                pers[i] = float(np.clip(l_ / e_, 0.0, 1.2))
            elif l_ > rate_floor[i]:
                pers[i] = 1.2      # деградация началась поздно и значима
        w = min(1.0, (len(cyc) - 1) / 12.0)   # стягивание при малых префиксах
        return 0.5 + (pers - 0.5) * w

    k_sei = consts.k_sei_m_s * (10.0 ** float(z[0]))
    d_solv = consts.d_solv_m2_s * (10.0 ** float(z[1]))
    j0_pl = consts.j0_pl_a_m2 * (10.0 ** float(z[2]))
    rho_sei = consts.rho_sei_ohm_m * (10.0 ** float(z[3]))
    area_eff = 10.0 ** float(z[4])
    a_sei = consts.alpha_sei * F_CONST / (R_CONST * T_REF)
    a_pl = consts.alpha_pl * F_CONST / (R_CONST * T_REF)
    gains = np.asarray(gains, float) if gains is not None \
        else _prefix_gains(lam_snr)
    pers = _prefix_persistence()
    beta_v = np.clip(
        trans.beta.detach().cpu().numpy(), -1.5, 0.5) \
        if trans is not None and hasattr(trans, "beta") \
        else np.zeros(4)

    q_li = res.q_li_ah[-1] if q_li_start is None else q_li_start
    # согласование скорости на границе истории (опц.; на коротком
    # префиксе ранняя быстрая деградация даёт систематический перегиб —
    # выключено по умолчанию после экспериментальной проверки)
    dql_obs = 0.0
    if len(res.q_li_ah) >= 4:
        m = min(8, len(res.q_li_ah))
        dql_obs = float(np.polyfit(
            np.asarray(res.cycles[-m:], float),
            np.asarray(res.q_li_ah[-m:], float), 1)[0])  # А·ч/цикл
    delta = consts.delta_sei0_m
    # омическая часть без плёнки неотрицательна (идентифицированное R
    # может быть шумовой оценкой ниже плёночного вклада)
    r_base = max(res.r_total_ohm[-1] - delta * rho_sei / area_m2, 0.0)
    cyc0 = res.cycles[-1]
    r_anchor = max(res.r_total_ohm[-1], 1e-3)
    qli_end = max(res.q_li_ah[-1] if q_li_start is None else q_li_start,
                  1e-6)
    dead = False
    rate_s = None
    cycles, soh_l, ql_l, r_l, d_l, qd_l, tn_l, tp_l, vh_l = \
        [], [], [], [], [], [], [], [], []
    qraw_l, ln_l, lp_l, qwin_l, qsim_l = [], [], [], [], []

    # метка прогнозной точки совпадает с возрастом состояния:
    # первый узел — сама граница истории (приращений ещё не было),
    # обновление состояния переносится в конец итерации
    k = cyc0
    while k <= cycle_end:
        kk = k
        if trans is None:
            lam_n = lam_at(lam_n0, dlam_n, kk)
            lam_p = lam_at(lam_p0, dlam_p, kk)
        win = resolve_windows_full(
            ocv_n, ocv_p, c_n, c_p, aff_n, aff_p, dn, dp,
            res.q_n_ah, res.rho, q_li, lam_n, lam_p, v_min, v_max)
        if win["degenerate"]:
            # запас лития не накрывает окно отсечек — элемент «умер»:
            # остаток горизонта — нулевая ёмкость
            dead = True
            break
        # R для симуляции: плёнка, а при гипотезе R(q_li) —
        # повышение обратно пропорционально потере инвентаря
        r_sim = r_base + delta * rho_sei / area_m2 \
            if (r_film or trans is None) else r_state
        if r_qli_gamma > 0:
            r_sim = max(r_sim, min(
                r_anchor * (qli_end / q_li) ** r_qli_gamma, 1e3))
        if r_fate is not None:
            # сценарий взрыва R: логистическая пола до r_max —
            # префикс не видит будущего роста (I·R < шума), судьба
            # задаётся популяционным приором извне
            r_f = min(r_fate[0]
                      / (1.0 + math.exp(-(kk - r_fate[1]) / r_fate[2])),
                      1e3)
            r_sim = max(r_sim, r_f)
        out = simulate_batch(
            t, i, mask, ocv_n, ocv_p,
            torch.tensor([win["theta_n0"]], dtype=torch.float64),
            torch.tensor([win["theta_p0"]], dtype=torch.float64),
            torch.tensor(res.q_n_ah, dtype=torch.float64),
            torch.tensor(res.rho, dtype=torch.float64),
            torch.tensor([r_sim], dtype=torch.float64),
            torch.tensor(j0n, dtype=torch.float64),
            torch.tensor(j0p, dtype=torch.float64),
            area_m2, c_n, c_p,
            lam_n=torch.tensor([lam_n], dtype=torch.float64),
            lam_p=torch.tensor([lam_p], dtype=torch.float64),
            ocv_p_affine=aff_p, ocv_n_affine=aff_n,
            ocv_p_delta=dp, ocv_n_delta=dn,
            edge_v=500.0 if ocv_clamp else 0.0)
        phi_n = out["phi_n"][0]
        w = torch.zeros(len(ii), dtype=torch.float64)
        dts = torch.diff(t[0])
        w[0], w[-1] = 0.5 * dts[0], 0.5 * dts[-1]
        w[1:-1] = 0.5 * (dts[:-1] + dts[1:])
        expo = torch.exp(a_sei * (phi_n - consts.u_sei_v)).clamp(max=1e6)
        j_sei = (-F_CONST * consts.c_solv_mol_m3
                 / (delta / d_solv + expo / k_sei)).abs()
        int_sei = float((j_sei * w).sum().detach()) * area_eff / 3600.0
        expp = torch.exp((a_pl * phi_n).clamp(max=50.0))
        expm = torch.exp((-a_pl * phi_n).clamp(max=50.0))
        j_pl = torch.where(phi_n < 0.0, j0_pl * (expp - expm),
                           torch.zeros_like(phi_n))
        int_pl = float((j_pl.abs() * w).sum().detach()) * area_eff / 3600.0
        dq_cycle = int_sei + consts.beta_dead * int_pl
        if rate_match and rate_s is None and dq_cycle > 0:
            # масштаб скорости: механика → наблюдаемый наклон префикса
            if dql_obs < 0:
                rate_s = float(np.clip(-dql_obs / dq_cycle, 0.05, 20.0))
            else:
                rate_s = 1.0
        if trans is not None and feats is not None:
            # переходная модель: g·Δq_mech + остаток, Δλ, Δlog R
            dyn = torch.tensor(
                [[q_li / qli0_, lam_n, lam_p,
                  math.log10(max(r_state / r0_, 1e-6)),
                  dq_cycle / qli0_, math.log10(kk + 1.0) / 4.0,
                  stride / 20.0, dq_prev]], dtype=torch.float64)
            with torch.no_grad():
                o = trans(dyn, feats.reshape(1, -1))[0]
            # поэлементная персистентность: эффективный показатель
            # затухания β_eff = β·(1−p); множитель (k/K)^(−β·p)
            # непрерывен при k=K и восстанавливает персистентный темп
            boost = ((kk + 1.0) / (cyc0 + 1.0)) ** (-beta_v * pers)
            dq_next = (float(o[0]) * dq_cycle / qli0_
                       + float(o[1])) * boost[0]
            dq_inc = stride * gains[0] * dq_next * qli0_ * (rate_s or 1.0)
            # инкременты физически пропорциональны остатку: за цикл
            # нельзя потерять >~0.5% инвентаря — иначе выученный
            # выброс даёт нефизичную ступеньку вместо спада
            dq_inc = float(np.clip(
                dq_inc, 0.0, q_li * (1.0 - math.exp(-5e-3 * stride))))
            dlamn_inc = float(np.clip(
                stride * gains[1] * boost[1] * float(o[2]),
                0.0, 0.01 * stride))
            dlamp_inc = float(np.clip(
                stride * gains[2] * boost[2] * float(o[3]),
                0.0, 0.01 * stride))
            # r_boost=1 — полная персистентность; 0 — выученный темп
            boost3 = 1.0 + r_boost * (boost[3] - 1.0)
            dlogr_inc = float(np.clip(
                stride * gains[3] * boost3 * float(o[4]),
                0.0, 0.01 * stride))
            r_tot = r_state
        else:
            dq_next = dq_prev
            dq_inc = float(np.clip(
                stride * dq_cycle * (rate_s or 1.0),
                0.0, q_li * (1.0 - math.exp(-5e-3 * stride))))
            dlamn_inc = dlamp_inc = dlogr_inc = 0.0
            r_tot = r_base + delta * rho_sei / area_m2
        ddelta_inc = stride * int_sei * 3600.0 * consts.v_sei_m3_mol \
            / (F_CONST * area_eff) * (rate_s or 1.0)
        if r_film and trans is not None:
            # плёночная компонента: рост только от SEI
            # (выученный ΔlogR-канал и экстраполяция префиксного
            # наклона экспериментально показали нестабильность)
            r_tot = r_base + delta * rho_sei / area_m2
        if r_qli_gamma > 0:
            # гипотеза: R растёт обратно пропорционально потере
            # инвентаря лития, R = R_гр·(q_li_гр/q_li)^γ —
            # темп привязан к выученной деградации конкретного
            # элемента и самонасыщается
            r_ql = r_anchor * (qli_end / q_li) ** r_qli_gamma
            r_tot = max(r_tot, min(r_ql, 1e3))
        if r_fate is not None:
            r_tot = max(r_tot, r_f)
        q_sim = np.nan
        if kinetic_cap:
            # кинетическая ёмкость: разрядный заряд, взвешенный
            # сигмоидой по (v−v_min)/ε — отсечка непрерывна по
            # состоянию (ε~15 мВ ~ шум измерения); убирает дискретный
            # перескок конца разряда между сегментами v(t)
            vv = out["v_hat"][0].detach().cpu().numpy()
            i_np = i[0].detach().cpu().numpy()
            t_np = t[0].detach().cpu().numpy()
            # клип аргумента: сигмоида насыщается задолго до ±50,
            # убирает переполнение exp без потери точности
            z = np.clip((vv - v_min) / 0.015, -50.0, 50.0)
            w = 1.0 / (1.0 + np.exp(-z))
            q_sim = float(np.trapezoid(
                -np.where(i_np < -1e-7, i_np, 0.0) * w, t_np)) / 3600.0
            # q_sim<=0 при живом окне — артефакт симуляции (недооценка
            # начального напряжения), не смерть: откат к окну
            if not np.isfinite(q_sim) or (q_sim <= 0.0
                                        and not win["degenerate"]):
                q_sim = np.inf
            # плавное слияние ограничений вместо жёсткого min:
            # реальное колено сглажено гетерогенностью элемента
            q_w = win["q_dch_ah"]
            tau = max(0.05 * q_w, 1e-9)
            q_lo = min(q_w, q_sim)
            q_dch = q_lo - tau * math.log(
                math.exp((q_lo - q_w) / tau)
                + math.exp((q_lo - q_sim) / tau))
        else:
            q_dch = win["q_dch_ah"]
        if kinetic_cap and q_dch <= 0.0:
            dead = True
            break
        if q_ref_ah is None:
            q_ref_ah = q_dch
        cycles.append(kk)
        soh_l.append(q_dch / q_ref_ah)
        ql_l.append(q_li)
        r_l.append(r_tot)
        d_l.append(delta)
        qd_l.append(q_dch)
        qraw_l.append(q_dch)
        tn_l.append(win["theta_n0"])
        tp_l.append(win["theta_p0"])
        ln_l.append(lam_n)
        lp_l.append(lam_p)
        qwin_l.append(win["q_dch_ah"])
        qsim_l.append(q_sim if kinetic_cap else np.nan)
        if save_curves:
            vh_l.append(out["v_hat"][0].detach().numpy().copy())
        # приращения переводят состояние kk → kk+stride: записанные
        # массивы описывают состояние на своей метке цикла
        dq_prev = dq_next
        lam_n = float(np.clip(lam_n - dlamn_inc, 0.4, 1.6))
        lam_p = float(np.clip(lam_p - dlamp_inc, 0.4, 1.6))
        # физический потолок: максимум по популяции ~1 кОм
        # (плёнка SEI + контакты); без ограничения ΔlogR-канал
        # экстраполируется экспоненциально и убивает разряд
        r_state = min(r_state * 10.0 ** dlogr_inc, 1e3)
        q_li = max(q_li - dq_inc, 1e-6)
        delta = max(delta + ddelta_inc, 0.0)
        k += stride

    if dead:
        # вырождение могло наступить на первом же прогнозном шаге —
        # тогда списки пусты; опорные значения — граничное состояние
        k_last = cycles[-1] if cycles else cyc0
        tail = np.arange(k_last + stride, cycle_end + 1, stride)
        # асимптотический хвост: остаточная ёмкость затухает
        # экспоненциально от последнего ЖИВОГО значения (отсчёт от
        # последнего живого цикла, не от конца хвоста), а не
        # обрывается вертикально в ноль
        tau = max(0.15 * (cycle_end - (cycles[0] if cycles else k_last)),
                  3.0 * stride)
        q_tail = ((qd_l[-1] if qd_l else 0.0)
                  * np.exp(-(tail - k_last - stride) / tau))
        cycles = cycles + tail.tolist()
        ql_l += [1e-6] * len(tail)
        qd_l += q_tail.tolist()
        qraw_l += q_tail.tolist()
        r_l += [r_l[-1] if r_l else res.r_total_ohm[-1]] * len(tail)
        d_l += [d_l[-1] if d_l else delta] * len(tail)
        tn_l += [tn_l[-1] if tn_l else res.theta_n0[-1]] * len(tail)
        tp_l += [tp_l[-1] if tp_l else res.theta_p0[-1]] * len(tail)
        ln_l += [ln_l[-1] if ln_l else lam_n] * len(tail)
        lp_l += [lp_l[-1] if lp_l else lam_p] * len(tail)
        qwin_l += [qwin_l[-1] if qwin_l else 0.0] * len(tail)
        qsim_l += [qsim_l[-1] if qsim_l else np.nan] * len(tail)

    if not qd_l:
        # пустой горизонт: граница истории — последний фактический цикл
        empty = np.empty(0)
        return ForecastResult(
            cycles=empty, soh=empty, q_li_ah=empty, r_total_ohm=empty,
            delta_sei_m=empty, q_dch_ah=empty, theta_n0=empty,
            theta_p0=empty, lam_n=empty, lam_p=empty,
            q_dch_raw=np.empty(0), q_win=np.empty(0), q_sim=np.empty(0),
            gains=gains, floors=pers)

    # деградация монотонна: проекция ёмкости на набор невозрастающих
    # значений убирает немонотонные артефакты экстраполяции λ/окон
    qd = np.minimum.accumulate(np.asarray(qd_l))
    if q_ref_ah is not None:
        soh_a = qd / q_ref_ah
    else:
        soh_a = qd / max(qd[0], 1e-12)
    return ForecastResult(
        cycles=np.asarray(cycles), soh=soh_a,
        q_li_ah=np.asarray(ql_l), r_total_ohm=np.asarray(r_l),
        delta_sei_m=np.asarray(d_l), q_dch_ah=qd,
        theta_n0=np.asarray(tn_l), theta_p0=np.asarray(tp_l),
        lam_n=np.asarray(ln_l), lam_p=np.asarray(lp_l),
        q_dch_raw=np.asarray(qraw_l),
        q_win=np.asarray(qwin_l), q_sim=np.asarray(qsim_l),
        v_hat=np.asarray(vh_l) if vh_l else None,
        gains=gains, floors=pers,
    )


def forecast_ensemble(
    res: IdentifiedCell,
    df,
    ocv_n: MonotoneOCV,
    ocv_p: MonotoneOCV,
    consts: DegradationConsts,
    z: torch.Tensor,
    n_members: int = 16,
    sigma_z: float = 0.4,
    seed: int = 0,
    **kw,
) -> dict:
    """Ансамбль прогнозов: z + шум σ (в декадах) → квантили SOH.

    Возвращает ``{"cycles", "soh_p10", "soh_p50", "soh_p90", "members"}``.
    Шум по лог-множителям покрывает только неопределённость кинетики z;
    неопределённость идентификации задаётся другими осями ансамбля
    (ident-ensemble, r_fate).
    """
    rng = np.random.default_rng(seed)
    members, cycs = [], []
    for _ in range(n_members):
        z_j = z + torch.randn(z.shape, dtype=torch.float64) * sigma_z \
            * torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5], dtype=torch.float64)
        fc = forecast(res, df, ocv_n, ocv_p, consts, z_j, **kw)
        members.append(fc.soh)
        cycs.append(fc.cycles)
    # выравнивание по общей сетке циклов (участники могут обрываться)
    k0 = min(c[0] for c in cycs)
    k1 = max(c[-1] for c in cycs)
    k_grid = np.arange(k0, k1 + 1, 1.0)
    sm = np.stack([np.interp(k_grid, c, m, left=np.nan, right=0.0)
                   for c, m in zip(cycs, members)])
    return {
        "cycles": k_grid,
        "soh_p10": np.nanpercentile(sm, 10, axis=0),
        "soh_p50": np.nanpercentile(sm, 50, axis=0),
        "soh_p90": np.nanpercentile(sm, 90, axis=0),
        "members": members,
    }
