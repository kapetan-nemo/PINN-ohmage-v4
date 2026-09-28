"""Сводка прогнозного теста: объединяет summary_<cell>.json в таблицу
и строит единую фигуру — подграфик на элемент, семейство прогнозов
по K на каждом.

    .venv/bin/python scripts/forecast_report.py [--tag run_...]

Без ``--tag`` берётся новейший подкаталог ``reports/forecast/<tag>``;
плоские артефакты (устаревший формат) читаются из корня каталога.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "forecast"


def pick_dir(tag: str | None) -> Path:
    if tag:
        return OUT / tag
    subs = [d for d in OUT.iterdir()
            if d.is_dir() and list(d.glob("summary_*.json"))]
    if subs:
        return max(subs, key=lambda d: d.stat().st_mtime)
    return OUT


def load_rows(d: Path):
    rows, cfgs = [], set()
    for p in sorted(d.glob("summary_*.json")):
        doc = json.loads(p.read_text())
        if isinstance(doc, dict):
            cfgs.add(json.dumps(doc.get("config", {}), sort_keys=True))
            rows += doc["rows"]
        else:
            rows += doc
    return rows, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    d = pick_dir(args.tag)
    print("каталог:", d)
    rows, cfgs = load_rows(d)
    if not rows:
        print("нет сводок"); return
    if cfgs:
        print("конфигурация прогонов:", *sorted(cfgs), sep="\n  ")
    Ks = sorted({r["K"] for r in rows})
    print(f"{'элемент':<22} K     идент  V RMSE,мВ  SOH RMSE  A-эмпир.  "
          f"k80 факт→прогноз (ошибка)")
    for r in sorted(rows, key=lambda r: (r["cell"], r["K"])):
        if "error" in r:
            print(f"{r['cell']:<22} K={r['K']:<3}  СБОЙ: {r['error'][:60]}")
            continue
        k80t = f"{r['k80_true']:.0f}" if r.get("k80_true") else "—"
        k80p = f"{r['k80_pred']:.0f}" if r.get("k80_pred") else "—"
        e80 = f"{r['k80_err']:+.0f}" if r.get("k80_err") is not None else "—"
        ra = f"{r['soh_rmse_empirical']:.3f}" if "soh_rmse_empirical" in r \
            else "—"
        print(f"{r['cell']:<22} K={r['K']:<3} {r['n_ident']:>5}  "
              f"{r['v_rmse_mv']:>9.0f}  {r['soh_rmse']:>8.3f}  {ra:>8}  "
              f"{k80t:>5}→{k80p:>5} ({e80})")
    print("\nсредний SOH RMSE по K (PINN | эмпирика A):")
    for K in Ks:
        v = [r["soh_rmse"] for r in rows if r.get("K") == K and "soh_rmse" in r]
        va = [r["soh_rmse_empirical"] for r in rows
              if r.get("K") == K and "soh_rmse_empirical" in r]
        if v:
            a_txt = f"{sum(va)/len(va):.3f}" if va else "—"
            print(f"  K={K:>3}: {sum(v)/len(v):.3f} | {a_txt} (n={len(v)})")

    # --- единая фигура: подграфик на элемент --------------------------
    cells = sorted({p.name.split("_K")[0] for p in d.glob("*_K*.npz")})
    if cells:
        cmap = plt.get_cmap("tab10")
        ncol = 3
        nrow = int(np.ceil(len(cells) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.4 * nrow),
                                 squeeze=False)
        ks_all = sorted({int(p.stem.split("_K")[1]) for c in cells
                         for p in d.glob(f"{c}_K*.npz")})
        kcol = {k: cmap(i % 10) for i, k in enumerate(ks_all)}
        for i, cid in enumerate(cells):
            ax = axes[i // ncol][i % ncol]
            for k in ks_all:
                f = d / f"{cid}_K{k}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                ax.plot(z["cycles_pred"], z["soh_pred"], "-", lw=1.6,
                        color=kcol[k], label=f"K={k}")
                # точка границы истории: последний измеренный цикл
                # перед началом прогноза
                ct, st = z["cycles_true"], z["soh_true"]
                k0 = z["cycles_pred"][0]
                m = ct < k0
                if m.any():
                    ax.plot(ct[m][-1], st[m][-1], "o", ms=7,
                            color=kcol[k], mec="k", mew=0.9, zorder=5)
            z0 = np.load(d / f"{cid}_K{ks_all[0]}.npz")
            ax.plot(z0["cycles_true"], z0["soh_true"], "k-", lw=2.6,
                    label="факт", zorder=4)
            for lv in (0.95, 0.90, 0.85, 0.80):
                ax.axhline(lv, color="r", ls=":", lw=0.6, alpha=0.4)
            ax.grid(True, which="both", ls="-", lw=0.4, alpha=0.5)
            ax.set_title(cid.replace("empa__", ""), fontsize=10)
            ax.set_ylim(-0.05, 1.1)
            if i // ncol == nrow - 1:
                ax.set_xlabel("цикл")
            if i % ncol == 0:
                ax.set_ylabel("SOH")
        handles = [plt.Line2D([], [], color=kcol[k], lw=1.6,
                            label=f"K={k}") for k in ks_all]
        handles.append(plt.Line2D([], [], color="k", lw=2.6, label="факт"))
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(handles), fontsize=9,
                   bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(f"прогноз SOH по первым K циклам — прогон «{d.name}»",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.98])
        fig.savefig(d / "forecast_grid.png", dpi=120)
        plt.close(fig)
        print("сохранена общая фигура:", d / "forecast_grid.png")


if __name__ == "__main__":
    main()
