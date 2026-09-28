"""Диагностика stage 1: пошаговый разбор невязки и дамп кривых v/v_hat.

Применение:
    .venv/bin/python scripts/stage1_diag.py --cell empa__ccid000002 --stride 20
    .venv/bin/python scripts/stage1_diag.py --cell empa__ccid000002 --anchor nmc811_Chen2020
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.data.bdf_loader import list_cells, load_bdf  # noqa: E402
from pinn_soh.data.metadata import parse_metadata  # noqa: E402
from pinn_soh.data.preprocess import build_pseudo_ocv, preprocess_cell  # noqa: E402
from pinn_soh.data.quality import build_cell_report  # noqa: E402
from pinn_soh.train.stage1_extract_state import (  # noqa: E402
    identify_cell, load_anchors, load_pretrained_ocv,
)

PROC = ROOT / "data" / "processed"


def classify_step(i_a: np.ndarray, i_nom: float) -> np.ndarray:
    """Классификация точек по току: chg_cc/chg_cv/dchg_cc/dchg_cv/rest."""
    out = np.full(len(i_a), "rest", dtype=object)
    chg = i_a > 1e-4
    dchg = i_a < -1e-4
    out[chg & (np.abs(i_a) > 0.5 * i_nom)] = "chg_cc"
    out[chg & (np.abs(i_a) <= 0.5 * i_nom)] = "chg_cv"
    out[dchg & (np.abs(i_a) > 0.5 * i_nom)] = "dchg_cc"
    out[dchg & (np.abs(i_a) <= 0.5 * i_nom)] = "dchg_cv"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="empa__ccid000002")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--iters-a", type=int, default=250)
    ap.add_argument("--iters-b", type=int, default=400)
    ap.add_argument("--anchor", default=None, help="принудительный якорь катода")
    ap.add_argument("--k", type=int, default=3, help="сколько циклов дампить в csv")
    args = ap.parse_args()

    cells_idx = {c.cell_id: c for c in list_cells(ROOT / "data" / "raw" / "aurora")}
    cf = cells_idx[args.cell]
    meta = parse_metadata(cf.metadata_path)
    df = load_bdf(cf.parquet_path)
    rep, cycles = build_cell_report(args.cell, df, meta)
    proc_path = PROC / f"{args.cell}.parquet"
    if proc_path.exists():
        pdf = pl.read_parquet(proc_path)
    else:
        proc = preprocess_cell(args.cell, df, cycles, rep.formation_cycles,
                               set(rep.artefact_cycles), rep.current_sign)
        proc.df.write_parquet(proc_path)
        pdf = proc.df

    area_m2 = (meta.electrode_areas.get("positive_cm2") or 1.539) * 1e-4
    ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
    anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")

    pseudo = build_pseudo_ocv(df, rep.formation_cycles)
    res = identify_cell(
        args.cell, pdf, cycles, ocv_n, ocv_p, anchors,
        formation_cycles=rep.formation_cycles, area_m2=area_m2,
        stride=args.stride, iters_a=args.iters_a, iters_b=args.iters_b,
        v_min=rep.protocol_summary.get("v_min_main") or 2.5,
        v_max=rep.protocol_summary.get("v_max_main") or 4.2,
        verbose=True, anchor=args.anchor, pseudo=pseudo,
    )
    print(f"RMSE итого: {res.v_rmse_mv:.1f} мВ | "
          f"affine катода: scale={res.ocv_p_scale:.4f} shift={res.ocv_p_shift_v*1000:.1f} мВ")

    out_dir = ROOT / "reports" / "stage1_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    v_hat = res.v_hat.numpy()
    i_nom = float(np.median(np.abs(pdf.filter(pl.col("i_a").abs() > 1e-4)["i_a"])))

    rows = []
    for bi, cid in enumerate(res.cycles):
        sub = pdf.filter(pl.col("cycle") == cid).sort("t_s")
        n = min(sub.height, v_hat.shape[1])
        idx = np.linspace(0, sub.height - 1, n).round().astype(int) if sub.height > n else np.arange(n)
        vv = sub["v_v"].to_numpy()[idx]
        ii = sub["i_a"].to_numpy()[idx]
        tt = sub["t_s"].to_numpy()[idx]
        st = classify_step(ii, i_nom)
        vh = v_hat[bi][:len(idx)]
        for s in ("chg_cc", "chg_cv", "dchg_cc", "dchg_cv", "rest"):
            sel = st == s
            if sel.sum() > 5:
                rows.append({"cycle": cid, "step": s,
                             "rmse_mv": float(np.sqrt(np.mean((vh[sel] - vv[sel]) ** 2)) * 1000),
                             "bias_mv": float(np.mean(vh[sel] - vv[sel]) * 1000),
                             "n": int(sel.sum())})
        if bi < args.k:
            extra = {}
            for k_, term in res.debug_terms.items():
                arr = term[bi].numpy()[:len(idx)]
                extra[k_] = arr
            pl.DataFrame({"t_s": tt, "step": st, "i_a": ii,
                          "v_meas": vv, "v_hat": vh, **extra}).write_csv(
                out_dir / f"{args.cell}_cycle{cid}.csv")
    rep_df = pl.DataFrame(rows)
    rep_df.write_csv(out_dir / f"{args.cell}_steps.csv")
    print(rep_df.group_by("step").agg(
        pl.col("rmse_mv").mean().round(1).alias("rmse_mv"),
        pl.col("bias_mv").mean().round(1).alias("bias_mv")))
    print(f"дамп → {out_dir}")


if __name__ == "__main__":
    main()
