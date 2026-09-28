"""Предобработка измерений для идентификации состояния и обучения.

Для каждого элемента:

* ток приводится к соглашению «заряд положителен»;
* поцикловые таблицы (из ``data/processed/cycles/``) дают границы,
  ёмкости и статус циклов; артефактные циклы исключаются из подгонки;
* временной ряд каждого основного цикла прореживается до ~``target_points``
  точек с сохранением фронтов (изменения тока и скачки напряжения сохраняются
  полностью, равномерная подсетка заполняет остальное);
* формовочные циклы (малый ток ~C/10) сохраняются отдельно — они
  используются для псевдоравновесной кривой и идентификации латентного
  кода ``c``;
* строится псевдоравновесная кривая элемента: точки OCV-подобия —
  хвосты пауз (I≈0, V≈const) и участки формовки, усреднённые по
  ёмкостной сетке.

Результат — ``data/processed/<cell>.parquet`` (прореженные ряды всех
циклов со столбцом ``segment``: ``formation|main|rest0``) и
``data/processed/pseudo_ocv/<cell>.parquet`` (``q_ah, v``).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

REST_CURRENT_A = 1e-4
V_JUMP_V = 0.02          # скачок напряжения — точка фронта, сохраняется
I_STEP_A = 2e-4          # изменение тока — точка фронта


def downsample_cycle(
    t: np.ndarray,
    i: np.ndarray,
    v: np.ndarray,
    target_points: int = 300,
) -> np.ndarray:
    """Индексы прореживания: все фронты + равномерная подсетка.

    Фронт — точка, где |ΔI| > I_STEP_A или |ΔV| > V_JUMP_V по соседним
    точкам. К ним добавляются равномерно распределённые точки до целевого
    числа. Возвращает отсортированные уникальные индексы.
    """
    n = len(t)
    if n <= target_points:
        return np.arange(n)
    di = np.abs(np.diff(i, prepend=i[0]))
    dv = np.abs(np.diff(v, prepend=v[0]))
    fronts = np.where((di > I_STEP_A) | (dv > V_JUMP_V))[0]
    # добавить соседей фронтов
    front_set = set(fronts.tolist())
    for f in fronts:
        front_set.update((f - 1, f + 1))
    grid = np.linspace(0, n - 1, target_points).round().astype(int)
    keep = np.array(sorted(front_set | set(grid.tolist())))
    keep = keep[(keep >= 0) & (keep < n)]
    return np.unique(keep)


@dataclass
class ProcessedCell:
    cell_id: str
    df: pl.DataFrame            # прореженные ряды всех циклов
    pseudo_ocv: pl.DataFrame    # псевдоравновесная кривая (q_ah, v)
    n_main_cycles: int
    formation_cycles: int


def build_pseudo_ocv(df: pl.DataFrame, formation_cycles: int,
                     v_bins: int = 300) -> pl.DataFrame:
    """Псевдоравновесная кривая из формовочных циклов (~C/10).

    Для каждого формовочного цикла строится средняя ветвь
    ``(V_зар(q) + V_раз(q))/2`` на общей сетке переданного заряда —
    омические и симметричные поляризационные члены сокращаются в
    первом порядке. По циклам — медиана. Возвращает ``(q_ah, v)``,
    где ``q_ah`` — заряд от начала формовочного цикла.

    Хвосты пауз основных циклов намеренно не используются: ось заряда
    от накопленного интеграла по всему испытанию смещена кулоновской
    утечкой (~0,03 %/цикл суммируется до долей мА·ч) и портит кривую.
    """
    mids = []
    q_ref_max = 0.0
    for cyc in df.filter(
            (pl.col("cycle") >= 1) & (pl.col("cycle") <= formation_cycles)
    )["cycle"].unique().sort():
        sub = df.filter(pl.col("cycle") == cyc).sort("t_s")
        if sub.height < 20:
            continue
        t = sub["t_s"].to_numpy()
        i = sub["i_a"].to_numpy()
        v = sub["v_v"].to_numpy()
        q = np.concatenate([[0.0], np.cumsum(i[:-1] * np.diff(t) / 3600.0)])
        q = q - q.min()
        chg = i > REST_CURRENT_A
        dch = i < -REST_CURRENT_A
        if chg.sum() < 10 or dch.sum() < 10:
            continue
        qc, qd = q[chg], q[dch]
        vc, vd = v[chg], v[dch]
        lo = max(qc.min(), qd.min()) + 0.02 * qc.max()
        hi = min(qc.max(), qd.max()) * 0.98
        if hi - lo < 0.3 * qc.max():
            continue
        grid = np.linspace(lo, hi, v_bins)
        v_mid = 0.5 * (np.interp(grid, qc, vc) + np.interp(grid, qd[::-1], vd[::-1]))
        mids.append((grid, v_mid))
        q_ref_max = max(q_ref_max, hi)
    if not mids:
        return pl.DataFrame({"q_ah": [], "v": []})
    # общая сетка: пересечение окон циклов
    lo = max(m[0][0] for m in mids)
    hi = min(m[0][-1] for m in mids)
    grid = np.linspace(lo, hi, v_bins)
    vs = np.stack([np.interp(grid, m[0], m[1]) for m in mids])
    return pl.DataFrame({"q_ah": grid, "v": np.median(vs, axis=0)})


def preprocess_cell(
    cell_id: str,
    df: pl.DataFrame,
    cycles: pl.DataFrame,
    formation_cycles: int,
    artefact_cycles: set[int] | None = None,
    current_sign: int = 1,
    target_points: int = 300,
) -> ProcessedCell:
    """Формирует прореженный набор циклов элемента и псевдо-OCV."""
    artefact_cycles = artefact_cycles or set()
    df = df.with_columns((pl.col("i_a") * current_sign).alias("i_a"))
    frames = []
    for row in cycles.iter_rows(named=True):
        cyc = row["cycle"]
        if cyc in artefact_cycles:
            continue
        seg = "formation" if 1 <= cyc <= formation_cycles else (
            "main" if cyc > formation_cycles else "rest0"
        )
        sub = df.filter(pl.col("cycle") == cyc).sort("t_s")
        if sub.height < 5:
            continue
        t = sub["t_s"].to_numpy()
        i = sub["i_a"].to_numpy()
        v = sub["v_v"].to_numpy()
        keep = downsample_cycle(t, i, v, target_points)
        frames.append(
            pl.DataFrame(
                {
                    "cycle": np.full(len(keep), cyc),
                    "t_s": t[keep],
                    "i_a": i[keep],
                    "v_v": v[keep],
                    "segment": [seg] * len(keep),
                }
            )
        )
    out = pl.concat(frames) if frames else pl.DataFrame()
    pseudo = build_pseudo_ocv(df, formation_cycles)
    return ProcessedCell(
        cell_id=cell_id,
        df=out,
        pseudo_ocv=pseudo,
        n_main_cycles=int(cycles.filter(pl.col("cycle") > formation_cycles).height),
        formation_cycles=formation_cycles,
    )
