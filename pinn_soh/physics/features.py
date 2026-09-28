"""Признаки циклов для идентификации и кодирующей сети.

Выделяются по измеренным рядам (t, I, V):

* положения и площади пиков dQ/dV — инкрементальная ёмкость
  ``|dQ/dV|``, вычисленная по разрядной ветке с фильтрацией
  Савицкого–Голея; пики соответствуют стадийным переходам электродов и
  используются для калибровки стехиометрических окон;
* омический скачок напряжения при изменении тока — мгновенный ΔV при
  переключении шага даёт оценку ``R_total``;
* амплитуда релаксации пауз — отклик напряжения за паузу;
* интервалы времени между порогами напряжения на заряде/разряде —
  интегральные характеристики формы кривой.
"""

import numpy as np
import polars as pl
from scipy.signal import find_peaks, savgol_filter


def dqdv_curve(v: np.ndarray, q_ah: np.ndarray, window: int = 31,
               poly: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Инкрементальная ёмкость |dQ/dV| по разрядной ветке.

    ``v`` — напряжение, ``q_ah`` — накопленная ёмкость от начала ветви.
    Сглаживание Савицкого–Голея по V; возвращает (v_smooth, |dQ/dV|).
    """
    if len(v) < window or window % 2 == 0:
        window = min(len(v) - (1 - len(v) % 2), max(5, window - 1))
    if len(v) < 5:
        return v, np.zeros_like(v)
    w = min(window, len(v) - 1 if len(v) % 2 == 0 else len(v))
    w = max(w - (1 - w % 2), 5)
    vs = savgol_filter(v, w, poly)
    qs = savgol_filter(q_ah, w, poly)
    dv = np.gradient(vs)
    dq = np.gradient(qs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where(np.abs(dv) > 1e-6, np.abs(dq / dv), 0.0)
    return vs, ic


def dqdv_peaks(v: np.ndarray, q_ah: np.ndarray, min_height_frac: float = 0.15,
               min_dist: int = 20) -> dict:
    """Положения пиков |dQ/dV| разрядной ветки (В) и их площади."""
    vs, ic = dqdv_curve(v, q_ah)
    if ic.max() <= 0:
        return {"peak_v": [], "peak_area_ah_per_v": [], "peak_positions_q": []}
    h = ic.max() * min_height_frac
    idx, props = find_peaks(ic, height=h, distance=min_dist)
    return {
        "peak_v": [float(vs[k]) for k in idx],
        "peak_height": [float(ic[k]) for k in idx],
        "peak_positions_q": [float(q_ah[k]) for k in idx],
    }


def ohmic_drop_estimate(df: pl.DataFrame, min_dt_s: float = 0.0,
                        max_dt_s: float = 45.0) -> float | None:
    """Оценка R_total по скачкам напряжения при изменении тока: ΔV/ΔI.

    Берутся переходы, где ток меняется резко (шаги протокола), а ΔV
    измеряется на первой точке после переключения.
    """
    t = df["t_s"].to_numpy()
    i = df["i_a"].to_numpy()
    v = df["v_v"].to_numpy()
    if len(t) < 10:
        return None
    di = np.abs(np.diff(i))
    jumps = np.where(di > 1e-4)[0]
    ratios = []
    for k in jumps:
        dt = t[k + 1] - t[k]
        if not (min_dt_s <= dt <= max_dt_s):
            continue
        dv = v[k + 1] - v[k]
        r = dv / di[k]
        if 0 < r < 10:
            ratios.append(abs(dv / di[k]))
    if not ratios:
        return None
    return float(np.median(ratios))


def relaxation_amplitude(df: pl.DataFrame, rest_current_a: float = 1e-4) -> float:
    """Средняя амплитуда релаксации напряжения в паузах (|ΔV| за паузу)."""
    i = df["i_a"].to_numpy()
    v = df["v_v"].to_numpy()
    rest = np.abs(i) < rest_current_a
    if rest.sum() < 5:
        return 0.0
    # границы пауз
    starts = np.where(np.diff(rest.astype(int)) == 1)[0] + 1
    amps = []
    for s in starts:
        e = s
        while e < len(v) and rest[e]:
            e += 1
        if e - s >= 5:
            amps.append(abs(v[e - 1] - v[s]))
    return float(np.mean(amps)) if amps else 0.0


def voltage_window_times(df: pl.DataFrame, levels=(3.6, 3.8, 4.0)) -> dict:
    """Время достижения порогов напряжения на заряде — интегральные
    характеристики формы кривой (доли длительности заряда)."""
    chg = df.filter(pl.col("i_a") > 1e-4).sort("t_s")
    if chg.height < 10:
        return {f"t_v_{lv}": np.nan for lv in levels}
    t = chg["t_s"].to_numpy()
    v = chg["v_v"].to_numpy()
    t0, t1 = t[0], t[-1]
    span = max(t1 - t0, 1.0)
    out = {}
    for lv in levels:
        hit = np.where(v >= lv)[0]
        out[f"t_v_{lv}"] = float((t[hit[0]] - t0) / span) if len(hit) else np.nan
    return out


def cycle_features(df: pl.DataFrame, formation_cycles: int = 3) -> dict:
    """Полный вектор признаков одного цикла (для кодирующей сети)."""
    i = df["i_a"].to_numpy()
    v = df["v_v"].to_numpy()
    t = df["t_s"].to_numpy()
    # разрядная ветвь: от максимума напряжения до конца цикла
    kmax = int(np.argmax(v)) if len(v) else 0
    dch = df.slice(kmax).filter(pl.col("i_a") < -1e-4).sort("t_s")
    peaks = {"peak_v": [], "peak_height": [], "peak_positions_q": []}
    if dch.height >= 20:
        td = dch["t_s"].to_numpy()
        vd = dch["v_v"].to_numpy()
        idc = dch["i_a"].to_numpy()
        q = np.concatenate([[0.0], np.cumsum(-idc[1:] * np.diff(td) / 3600.0)])
        peaks = dqdv_peaks(vd, q)
    out = {
        "r_ohm_est": ohmic_drop_estimate(df),
        "relax_amp_v": relaxation_amplitude(df),
        "n_dqdv_peaks": len(peaks["peak_v"]),
        "peak_v": peaks["peak_v"],
        "peak_q": peaks["peak_positions_q"],
    }
    out.update(voltage_window_times(df))
    return out
