"""Популяционное сравнение двух наборов идентификаций stage1.

Читает ``checkpoints/stage1_prev`` (эталон до выравнивания кода) и
``checkpoints/stage1`` (пересчёт текущим кодом). Для каждого элемента и
каждого канала состояния вычисляет медиану абсолютного расхождения
траекторий на общей сетке циклов, а также сравнивает невязку
напряжения ``v_rmse_mv``.

Вывод: сводная таблица квантилей по популяции и гистограмма
``checkpoints/compare_stage1.png``.

    .venv/bin/python scripts/compare_stage1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CHANNELS = [
    ("theta_n0", "θ_n0", ""),
    ("theta_p0", "θ_p0", ""),
    ("r_total_ohm", "R", "Ом"),
    ("q_li_ah", "q_li", "А·ч"),
    ("lam_n", "λ_n", ""),
    ("lam_p", "λ_p", ""),
]


def med_diff(a: dict, b: dict, key: str) -> float:
    ca, cb = np.asarray(a["cycles"]), np.asarray(b["cycles"])
    va, vb = np.asarray(a[key], float), np.asarray(b[key], float)
    grid = np.intersect1d(ca, cb)
    if len(grid) < 3:
        return np.nan
    ia = np.interp(grid, ca, va)
    ib = np.interp(grid, cb, vb)
    return float(np.median(np.abs(ia - ib)))


def main() -> None:
    prev = ROOT / "checkpoints" / "stage1_prev"
    new = ROOT / "checkpoints" / "stage1"
    diffs: dict[str, list] = {k: [] for k, _, _ in CHANNELS}
    rmse_old, rmse_new = [], []
    n_cells = 0
    for p_old in sorted(prev.glob("*.json")):
        p_new = new / p_old.name
        if not p_new.exists():
            continue
        a, b = json.loads(p_old.read_text()), json.loads(p_new.read_text())
        if not np.intersect1d(a["cycles"], b["cycles"]).size:
            continue
        n_cells += 1
        rmse_old.append(a["v_rmse_mv"])
        rmse_new.append(b["v_rmse_mv"])
        for k, _, _ in CHANNELS:
            diffs[k].append(med_diff(a, b, k))

    print(f"элементов сравнено: {n_cells}")
    print(f"V RMSE: медиана было {np.median(rmse_old):.1f} мВ → "
          f"{np.median(rmse_new):.1f} мВ "
          f"(p90 {np.percentile(rmse_old,90):.0f} → "
          f"{np.percentile(rmse_new,90):.0f})")
    print(f"{'канал':10} {'p50':>10} {'p90':>10} {'max':>10}")
    for k, name, unit in CHANNELS:
        d = np.asarray(diffs[k])
        d = d[np.isfinite(d)]
        print(f"{name:10} {np.median(d):10.3g} {np.percentile(d,90):10.3g} "
              f"{d.max():10.3g} {unit}")

    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    for ax, (k, name, unit) in zip(axes.ravel(), CHANNELS):
        d = np.asarray(diffs[k])
        ax.hist(d[np.isfinite(d)], bins=40, color="tab:blue", alpha=0.75)
        ax.set_title(f"|Δ {name}| медиана по траектории"
                     + (f" [{unit}]" if unit else ""))
        ax.set_yscale("log")
        ax.grid(True, which="both", lw=0.4, alpha=0.6)
    # диагональная диаграмма невязок V: старый vs выровненный код
    ax = axes.ravel()[6]
    ro, rn = np.asarray(rmse_old), np.asarray(rmse_new)
    lim = max(ro.max(), rn.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.scatter(ro, rn, s=14, alpha=0.6, color="tab:red")
    ax.set_xlabel("V RMSE старый, мВ")
    ax.set_ylabel("V RMSE выровненный, мВ")
    ax.set_title("невязка напряжения по элементам")
    ax.grid(True, which="both", lw=0.4, alpha=0.6)
    axes.ravel()[7].axis("off")
    fig.suptitle("расхождение идентификаций: старый код vs выровненный")
    fig.tight_layout()
    out = ROOT / "checkpoints" / "compare_stage1.png"
    fig.savefig(out, dpi=120)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
