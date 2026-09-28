"""Инженерный интерфейс прогноза: первые K циклов → когда элемент умрёт.

    .venv/bin/python scripts/predict_cell.py \
        --cell empa__ccid000208 --history 50 --horizon 800 \
        --levels 0.95 0.9 0.85 0.8 --save-curves --out report.json

Конвейер: загрузка → предобработка → идентификация по префиксу ≤K →
калибровка кинетики → рекурсивный прогноз на ``horizon`` циклов →
отчёт по порогам SOH. Возвращает JSON со всеми внутренними
состояниями (q_li, δ_SEI, R, окна θ) и опционально кривыми напряжения.
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
from pinn_soh.eval.levels import crossing_report  # noqa: E402
from pinn_soh.physics.degradation import DegradationConsts  # noqa: E402
from pinn_soh.predict.forecast import fit_z_prefix, forecast  # noqa: E402
from pinn_soh.train.stage1_extract_state import (  # noqa: E402
    identify_cell, load_anchors, load_pretrained_ocv, refine_per_cycle,
)

PROC = ROOT / "data" / "processed"


def predict_cell(cell_id: str, history: int, horizon: int,
                 levels=(0.95, 0.9, 0.85, 0.8), refine_iters: int = 150,
                 iters_b: int = 400, save_curves: bool = False,
                 stride: int = 5) -> dict:
    """Полный прогноз по первым ``history`` основных циклов."""
    t0 = time.time()
    cf = {c.cell_id: c for c in list_cells(ROOT / "data" / "raw" / "aurora")}[cell_id]
    meta = parse_metadata(cf.metadata_path)
    df = load_bdf(cf.parquet_path)
    rep, cycles = build_cell_report(cell_id, df, meta)
    proc_path = PROC / f"{cell_id}.parquet"
    if proc_path.exists():
        pdf = pl.read_parquet(proc_path)
    else:
        proc = preprocess_cell(cell_id, df, cycles, rep.formation_cycles,
                               set(rep.artefact_cycles), rep.current_sign)
        proc.df.write_parquet(proc_path)
        pdf = proc.df
    area_m2 = (meta.electrode_areas.get("positive_cm2") or 1.539) * 1e-4
    v_max = rep.protocol_summary.get("v_max_main") or 4.2
    v_min = rep.protocol_summary.get("v_min_main") or 2.5
    pseudo = build_pseudo_ocv(df, rep.formation_cycles)
    ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
    anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")
    kin = ROOT / "data" / "params" / "kinetics_OKane2022.json"
    consts = DegradationConsts.from_json(kin) if kin.exists() else DegradationConsts()

    first_main = rep.formation_cycles + 1
    last_obs = first_main + history - 1
    res = identify_cell(
        cell_id, pdf, cycles, ocv_n, ocv_p, anchors,
        formation_cycles=rep.formation_cycles, area_m2=area_m2, stride=20,
        iters_a=200, iters_b=iters_b, v_min=v_min, v_max=v_max,
        pseudo=pseudo, max_cycle=last_obs)
    if refine_iters:
        res = refine_per_cycle(res, pdf, cycles, ocv_n, ocv_p,
                               area_m2=area_m2, iters=refine_iters)
    z = fit_z_prefix(res, pdf, consts, iters=150)
    cycle_end = int(res.cycles[-1] + horizon)
    fc = forecast(res, pdf, ocv_n, ocv_p, consts, z, cycle_end=cycle_end,
                  v_min=v_min, v_max=v_max, stride=stride,
                  area_m2=area_m2, save_curves=save_curves)

    # измеренная SOH для контекста (не используется в прогнозе)
    cc = cycles.sort("cycle")
    ids = cc["cycle"].to_numpy(); q = cc["q_dchg_ah"].to_numpy()
    m = ids >= first_main
    soh_meas = q[m] / q[m][0]

    rep_lv = crossing_report(fc.cycles, fc.soh, levels=levels)
    return {
        "cell_id": cell_id,
        "history_cycles": history,
        "horizon_cycles": horizon,
        "v_max": v_max, "v_min": v_min,
        "identified": {
            "anchor": res.anchor_name,
            "q_n_ah": res.q_n_ah, "rho": res.rho,
            "v_rmse_mv": res.v_rmse_mv,
            "n_cycles_used": len(res.cycles),
            "last_observed_cycle": int(res.cycles[-1]),
            "q_li_ah": res.q_li_ah, "r_total_ohm": res.r_total_ohm,
            "theta_n0": res.theta_n0, "theta_p0": res.theta_p0,
            "lam_n": res.lam_n, "lam_p": res.lam_p,
        },
        "z_multipliers": [float(x) for x in z],
        "forecast": {
            "cycles": fc.cycles.tolist(),
            "soh": fc.soh.tolist(),
            "q_li_ah": fc.q_li_ah.tolist(),
            "r_total_ohm": fc.r_total_ohm.tolist(),
            "delta_sei_nm": (fc.delta_sei_m * 1e9).tolist(),
            "q_dch_ah": fc.q_dch_ah.tolist(),
        },
        "thresholds": rep_lv,
        "measured_cycles": ids[m].tolist(),
        "measured_soh": soh_meas.tolist(),
        "runtime_s": time.time() - t0,
        "_curves": {"v_hat": fc.v_hat.tolist() if fc.v_hat is not None else None,
                    "i_template_cycles": res.cycles[-1]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--history", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.95, 0.9, 0.85, 0.8])
    ap.add_argument("--refine", type=int, default=150)
    ap.add_argument("--iters-b", type=int, default=400)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--save-curves", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = predict_cell(args.cell, args.history, args.horizon,
                       levels=tuple(args.levels), refine_iters=args.refine,
                       iters_b=args.iters_b, save_curves=args.save_curves,
                       stride=args.stride)
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(txt)
        print("сохранено:", args.out)
    else:
        print(txt)


if __name__ == "__main__":
    main()
