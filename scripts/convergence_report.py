#!/usr/bin/env python
"""Анализ сходимости прогноза по длине префикса K.

Отдельно от целевой метрики (K=100). Собирает summary_<cell>.json в
каталоге прогона и проверяет: растёт ли качество при добавлении истории.

Метрики:
  * mean/median SOH RMSE и |k80_err| по каждому K;
  * доля элементов, где большее K строго лучше меньшего (матрица);
  * корреляция Спирмена log K vs ошибка (по всем парам элемент×K);
  * вердикт монотонности: каждое K лучше всех меньших по среднему RMSE.

Запуск:  python scripts/convergence_report.py reports/forecast/trans_v2
"""
import glob
import json
import os
import sys

import numpy as np


def _rho(x, y):
    """Корреляция Спирмена без scipy."""
    def rank(a):
        a = np.asarray(a, float); order = a.argsort(); r = np.empty(len(a))
        r[order] = np.arange(len(a))
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "summary_*.json")))
    rows = []
    for f in files:
        rows += json.load(open(f))["rows"]
    ks = sorted({r["K"] for r in rows})
    cells = sorted({r["cell"] for r in rows})
    tab = {(r["cell"], r["K"]): r for r in rows}

    print(f"прогон {os.path.basename(run_dir)}: {len(cells)} элементов, K={ks}\n")
    print(f"{'K':>5} | {'mean RMSE':>9} | {'med RMSE':>8} | {'mean|k80|':>9}"
          f" | {'med|k80|':>8} | n")
    print("-" * 55)
    mean_rmse, mean_k80 = {}, {}
    for k in ks:
        rr = [tab[(c, k)] for c in cells if (c, k) in tab]
        e = [r["soh_rmse"] for r in rr]
        ke = [abs(r["k80_err"]) for r in rr
              if r.get("k80_err") is not None
              and np.isfinite(r["k80_err"])]
        mean_rmse[k] = float(np.mean(e)); mean_k80[k] = float(np.mean(ke))
        print(f"{k:>5} | {np.mean(e):>9.3f} | {np.median(e):>8.3f}"
              f" | {np.mean(ke):>9.1f} | {np.median(ke):>8.1f} | {len(rr)}")

    print("\nпопарная матрица: доля элементов, где K(строка) лучше K(столбец)")
    print(f"{'':>5}" + "".join(f"{k:>7}" for k in ks))
    pair = {}
    for ki in ks:
        line = f"{ki:>5}"
        for kj in ks:
            if ki == kj:
                line += "      ."; continue
            wins, n = 0, 0
            for c in cells:
                a, b = tab.get((c, ki)), tab.get((c, kj))
                if a and b:
                    wins += a["soh_rmse"] < b["soh_rmse"]; n += 1
            pair[(ki, kj)] = wins / n if n else np.nan
            line += f"{wins}/{n:>3} "
        print(line)

    xs = [np.log(r["K"]) for r in rows]
    ys = [r["soh_rmse"] for r in rows]
    print(f"\nСпирмен (log K, RMSE): {_rho(xs, ys):+.3f}"
          "  (отрицательная = история помогает)")

    # монотонность: каждое K должно быть лучше ВСЕХ меньших
    print("\nпроверка монотонности (по среднему RMSE):")
    ok_all = True
    for i, ki in enumerate(ks):
        worse_than = [kj for kj in ks[:i] if mean_rmse[ki] >= mean_rmse[kj]]
        if worse_than:
            ok_all = False
            print(f"  K={ki}: ХУЖЕ чем {worse_than} ({mean_rmse[ki]:.3f})")
        elif i:
            print(f"  K={ki}: лучше всех меньших ({mean_rmse[ki]:.3f})")
    verdict = ("СХОДИТСЯ строго монотонно" if ok_all
               else "сходимость НЕСТРОГАЯ — улучшение в среднем есть, "
                    "но отдельные K регрессируют")
    print(f"\nВЕРДИКТ: {verdict}")

    out = {"run": os.path.basename(run_dir), "Ks": ks, "cells": cells,
           "mean_rmse": mean_rmse, "mean_abs_k80_err": mean_k80,
           "pairwise_win_frac": {f"{a}>{b}": v for (a, b), v in pair.items()},
           "spearman_logK_rmse": _rho(xs, ys), "strict_monotone": ok_all}
    with open(os.path.join(run_dir, "convergence.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"сохранено: {run_dir}/convergence.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/forecast")
