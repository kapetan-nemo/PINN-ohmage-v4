"""Метрики качества прогноза деградации и напряжения."""
from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def soh_rmse_by_horizon(pred_cycles: np.ndarray, pred_soh: np.ndarray,
                        true_cycles: np.ndarray, true_soh: np.ndarray,
                        horizons=(50, 100, 200, 500, 1000)) -> dict[int, float]:
    """RMSE SOH на горизонтах (число циклов от последнего обученного).

    ``pred_*`` — прогнозная траектория, ``true_*`` — измеренная;
    истинная SOH на цикле прогноза интерполируется по измеренной кривой.
    """
    out = {}
    for h in horizons:
        m = pred_cycles <= pred_cycles[0] + h
        if m.sum() < 2:
            out[h] = np.nan
            continue
        tt = np.interp(pred_cycles[m], true_cycles, true_soh)
        out[h] = rmse(pred_soh[m], tt)
    return out


def threshold_cycle(cycles: np.ndarray, soh: np.ndarray,
                    level: float) -> float | None:
    """Первый цикл, где SOH опускается ниже порога (линейная интерполяция)."""
    cycles = np.asarray(cycles, float)
    soh = np.asarray(soh, float)
    below = np.where(soh < level)[0]
    if len(below) == 0:
        return None
    k = int(below[0])
    if k == 0:
        return float(cycles[0])
    x0, x1 = cycles[k - 1], cycles[k]
    y0, y1 = soh[k - 1], soh[k]
    dy = y1 - y0
    if abs(dy) < 1e-12:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / dy)


def threshold_errors(pred_cycles: np.ndarray, pred_soh: np.ndarray,
                     true_cycles: np.ndarray, true_soh: np.ndarray,
                     levels=(0.95, 0.90, 0.85, 0.80)) -> dict[float, float | None]:
    """Ошибка (в циклах) определения пересечения порога SOH."""
    out = {}
    for lv in levels:
        kp = threshold_cycle(pred_cycles, pred_soh, lv)
        kt = threshold_cycle(true_cycles, true_soh, lv)
        out[lv] = (kp - kt) if (kp is not None and kt is not None) else None
    return out


def interval_coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    """Доля истинных значений, покрытых интервалом [lo, hi]."""
    return float(np.mean((y >= lo) & (y <= hi)))
