"""Попанельное сравнение прогноза состояния с полножизненной идентификацией.

Для каждого <cell>_K<K>.npz в каталоге прогона строит один файл
``<cell>_state_channels.png``: по панели на канал состояния
(SOH, q_li, R, λ_n, λ_p, θ_n0, θ_p0, δ_SEI, q_dch) —
точечный прогноз и медиана/полоса ансамбля против эталона
(полножизненная идентификация checkpoints/stage1 или измеренная SOH).
RMSE точечного и медианного прогноза — в заголовке панели.

    .venv/bin/python scripts/plot_state_forecast.py \
        --dir reports/forecast/audit_ce --cells empa__ccid000110
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_truth(cid: str) -> dict:
    """Эталонные траектории: полножизненная идентификация stage1."""
    p = ROOT / "checkpoints" / "stage1" / f"{cid}.json"
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    return {k: np.asarray(j[k], float) for k in
            ("cycles", "theta_n0", "theta_p0", "r_total_ohm", "q_li_ah",
             "lam_n", "lam_p", "hyst_mv")}


def measured_q_dchg(cid: str):
    """Измеренная разрядная ёмкость по циклам (без артефактных)."""
    from pinn_soh.data.bdf_loader import list_cells, load_bdf, detect_current_sign
    from pinn_soh.data.quality import per_cycle_stats
    for ds in (ROOT / "data" / "raw").iterdir():
        cells = {c.cell_id: c for c in list_cells(ds)} if ds.is_dir() else {}
        if cid in cells:
            df = load_bdf(cells[cid].parquet_path)
            cs = detect_current_sign(df)
            cyc = per_cycle_stats(df, cs).sort("cycle")
            cyc = cyc.filter(pl.col("q_dchg_ah") > 0)  # выдержка cy0, обрезки
            return cyc["cycle"].to_numpy(), cyc["q_dchg_ah"].to_numpy()
    return None, None


def rmse_at(grid, y, tx, ty):
    """RMSE y на сетке grid против истины (tx, ty) — только живые точки."""
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    yt = np.interp(grid[ok], tx, ty)
    return float(np.sqrt(np.mean((y[ok] - yt) ** 2)))


def plot_cell(npz_path: Path, out_png: Path):
    d = np.load(npz_path)
    cid = npz_path.name.split("_K")[0]
    tru = load_truth(cid)
    c, s = d["cycles_pred"], d["soh_pred"]
    cy_t, soh_t = d["cycles_true"], d["soh_true"]

    ens = None
    if "soh_ens_median" in d.files:
        ens = (d["cycles_ens"], d["soh_ens_median"],
               d["soh_ens_p10"], d["soh_ens_p90"])

    # панели: (ключ прогноза, ключ истины, имя, единицы)
    panels = [
        ("soh_pred", "soh_true", "SOH", ""),
        ("q_li", "q_li_ah", "инвентарь лития", "А·ч"),
        ("r_total", "r_total_ohm", "сопротивление", "Ом"),
        ("lam_n", "lam_n", "λ анода", ""),
        ("lam_p", "lam_p", "λ катода", ""),
        ("theta_n0", "theta_n0", "θ_n0", ""),
        ("theta_p0", "theta_p0", "θ_p0", ""),
        ("delta_sei", None, "δ SEI", "м"),
        ("q_dch", "q_dchg", "разрядная ёмкость", "А·ч"),
    ]
    qm_c, qm_v = measured_q_dchg(cid)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for ax, (pk, tk, name, unit) in zip(axes.ravel(), panels):
        if pk == "soh_pred":
            y, ty_ = s, (cy_t, soh_t)
        elif pk == "q_dch":
            y, ty_ = d["q_dch"], (qm_c, qm_v)
        elif tk and tk in tru:
            y, ty_ = d[pk], (tru["cycles"], tru[tk])
        else:
            y, ty_ = d[pk], None
        if ty_ is not None and ty_[0] is not None:
            ax.plot(ty_[0], ty_[1], "k-", lw=2.2, label="факт/идент.")
            rmse_p = rmse_at(c, y, *ty_)
        else:
            rmse_p = np.nan
        if ens is not None and pk == "soh_pred":
            kg, med, p10, p90 = ens
            ax.fill_between(kg, p10, p90, color="tab:blue",
                            alpha=0.15, lw=0)
            ax.plot(kg, med, ":", color="tab:blue", lw=2, label="медиана")
            rmse_m = rmse_at(kg, med, *ty_) if ty_ else np.nan
            ax.plot(kg, d["soh_ens_members"].T, "-", color="tab:blue",
                    lw=0.5, alpha=0.25)
        else:
            rmse_m = np.nan
        ax.plot(c, y, "-", color="tab:orange", lw=1.8, label="прогноз")
        ax.axvline(c[0], color="gray", ls=":", lw=1)
        ttl = name + (f" [{unit}]" if unit else "")
        ttl += f" | RMSE {rmse_p:.3g}"
        if np.isfinite(rmse_m):
            ttl += f" / мед {rmse_m:.3g}"
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, which="both", ls="-", lw=0.4, alpha=0.6)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{cid}: каналы прогноза против полножизненной правды")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("сохранено:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()
    tag = Path(args.dir)
    for npz in sorted(tag.glob("*_K*.npz")):
        cid = npz.name.split("_K")[0]
        if args.cells and cid not in args.cells:
            continue
        plot_cell(npz, tag / f"{cid}_state_channels.png")


if __name__ == "__main__":
    main()
