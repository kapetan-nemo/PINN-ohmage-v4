#!/usr/bin/env python
"""Обзор измеренных SOH всех элементов: есть ли в данных колено.

Фигуры:
  all_soh_overlay.png — все кривые на одних осях, цвет по группе;
  all_soh_panels.png  — сетка мини-панелей по ≤4 элемента;
  knee_detector: двухсегментная линейная аппроксимация, колено =
  breakpoint, где наклон растёт по модулю ≥2× и улучшает SSE ≥30%.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CYC = ROOT / "data" / "processed" / "cycles"
OUT = ROOT / "reports" / "forecast"
QUAL = json.loads((ROOT / "configs" / "cell_quality.json").read_text())


def load_all():
    out = {}
    for f in sorted(CYC.glob("*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["cycle", "q_dchg_ah"])
        except Exception:
            continue
        soh = d["q_dchg_ah"].to_numpy(float)
        q0 = np.nanmedian(soh[:10])
        out[f.stem] = (d["cycle"].to_numpy(float), soh / q0)
    return out


def knee_metric(k, y):
    """Возвращает (есть_колено, позиция, k1, k2): двухломаная подгонка."""
    n = len(k)
    if n < 40:
        return False, np.nan, 0.0, 0.0
    best = (np.inf, None)
    x = k - k[0]
    for i in range(int(0.3 * n), int(0.85 * n)):
        c1 = np.polyfit(x[:i], y[:i], 1)
        c2 = np.polyfit(x[i:], y[i:], 1)
        sse = np.sum((y[:i] - np.polyval(c1, x[:i])) ** 2) \
            + np.sum((y[i:] - np.polyval(c2, x[i:])) ** 2)
        if sse < best[0]:
            best = (sse, (i, c1, c2))
    i, c1, c2 = best[1]
    c0 = np.polyfit(x, y, 1)
    sse1 = np.sum((y - np.polyval(c0, x)) ** 2)
    # колено: второй наклон заметно круче + двухломаная лучше прямой
    has = (c2[0] < 1.6 * c1[0]) and (best[0] < 0.7 * sse1) and c1[0] < 0
    return has, float(k[i]), float(c1[0]), float(c2[0])


def main():
    data = load_all()
    cells = sorted(data)
    # --- оверлей -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.4))
    groups = {}
    for cid in cells:
        g = QUAL.get(cid, {}).get("chemistry_group", "?")
        vm = QUAL.get(cid, {}).get("v_max_main", 0)
        groups.setdefault(f"{g} {vm}V", []).append(cid)
    cmap = plt.get_cmap("tab10")
    for gi, (gname, ids) in enumerate(sorted(groups.items())):
        for cid in ids:
            k, y = data[cid]
            ax.plot(k, y, lw=0.7, alpha=0.45, color=cmap(gi % 10))
        ax.plot([], [], color=cmap(gi % 10), lw=1.5,
                label=f"{gname} (n={len(ids)})")
    ax.axhline(0.8, color="red", lw=0.8, ls="--", alpha=0.6)
    ax.set(xlabel="цикл", ylabel="SOH", ylim=(0, 1.05))
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("измеренная SOH — все элементы, по группам протоколов")
    fig.tight_layout()
    fig.savefig(OUT / "all_soh_overlay.png", dpi=130)

    # --- детектор колена -----------------------------------------------
    knees = []
    for cid in cells:
        k, y = data[cid]
        has, kb, s1, s2 = knee_metric(k, y)
        knees.append((cid, has, kb, s1, s2))
    n_knee = sum(k[1] for k in knees)
    print(f"элементов: {len(cells)}; с детектируемым коленом: {n_knee} "
          f"({100*n_knee/len(cells):.0f}%)")
    for cid, has, kb, s1, s2 in knees:
        if has:
            print(f"  {cid}: knee@{kb:.0f}  slope {s1*1e3:.2f}→"
                  f"{s2*1e3:.2f} /1kц")

    # --- производная SOH: метрика артефактов и те же панели -------------
    noise = {}
    for cid in cells:
        k, y = data[cid]
        ok = np.isfinite(k) & np.isfinite(y)
        k, y = k[ok], y[ok]
        if len(k) < 3:
            noise[cid] = np.zeros(len(k)); data[cid] = (k, y); continue
        dy = np.diff(y) / np.maximum(np.diff(k), 1.0)
        dy = np.concatenate([dy, [dy[-1]]])
        noise[cid] = dy; data[cid] = (k, y)
    njump = {c: int((np.abs(noise[c]) > 0.003).sum()) for c in cells}
    worst = sorted(njump.items(), key=lambda x: -x[1])[:15]
    print("\nтоп по |dSOH/dk|>0,003 скачков/цикл:", *worst[:15], sep="\n  ")

    per, ncol = 4, 4
    nrow = int(np.ceil(len(cells) / (per * ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow),
                             squeeze=False)
    cmap2 = plt.get_cmap("Set1")
    for i, ax in enumerate(axes.flat):
        ax.grid(True, alpha=0.3)
        chunk = cells[i * per:(i + 1) * per]
        if not chunk:
            ax.axis("off"); continue
        for j, cid in enumerate(chunk):
            k, y = data[cid]
            dy = pd.Series(noise[cid]).rolling(5, center=True,
                                               min_periods=1).median()
            ax.plot(k, dy, lw=0.9, color=cmap2(j % 9),
                    label=cid.replace("empa__ccid", "#"))
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=6, loc="lower left")
        ax.tick_params(labelsize=6)
    fig.suptitle("dSOH/dцикл (медианно сглажено, окно 5) — все элементы; "
                 "всплески ± = артефакты измерения")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / "all_dsoh_panels.png", dpi=120)

    # --- мини-панели по 4 элемента --------------------------------------
    knee_at = {c: kb for c, h, kb, *_ in knees if h}
    cells_plot = cells[:]
    nrow = int(np.ceil(len(cells_plot) / (per * ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow),
                             squeeze=False)
    cmap2 = plt.get_cmap("Set1")
    for i, ax in enumerate(axes.flat):
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        chunk = cells_plot[i * per:(i + 1) * per]
        if not chunk:
            ax.axis("off"); continue
        for j, cid in enumerate(chunk):
            k, y = data[cid]
            ax.plot(k, y, lw=1.0, color=cmap2(j % 9),
                    label=cid.replace("empa__ccid", "#"))
            if cid in knee_at:
                ax.axvline(knee_at[cid], color=cmap2(j % 9), ls=":",
                           lw=0.8, alpha=0.7)
        ax.axhline(0.8, color="red", lw=0.5, ls="--", alpha=0.5)
        ax.legend(fontsize=6, loc="lower left")
        ax.tick_params(labelsize=6)
    fig.suptitle("измеренная SOH — все элементы; пунктирные вертикали — "
                 "детектированное колено; красный пунктир — SOH 0.8")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / "all_soh_panels.png", dpi=120)
    print("сохранено:", OUT / "all_soh_overlay.png",
          OUT / "all_soh_panels.png", sep="\n  ")


if __name__ == "__main__":
    main()
