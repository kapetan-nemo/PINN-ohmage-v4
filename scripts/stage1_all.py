"""Полный проход идентификации состояния по обучаемым элементам.

Для каждого элемента из ``configs/cell_quality.json`` (статусы ok и
early_failure) выполняет предобработку (кэш ``data/processed/``),
идентификацию и сохраняет:

* ``checkpoints/stage1/<cell>.json`` — константы и поцикловые параметры;
* ``checkpoints/stage1/<cell>.npz`` — траектории (t, i, φ_n) и
  ``θ_n(t), θ_p(t)`` идентифицированных циклов для обучения деградации.

Запуск по шардам:
    .venv/bin/python scripts/stage1_all.py --shard 0/4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
    identify_cell, load_anchors, load_pretrained_ocv, pack_cycles,
    refine_per_cycle,
)

PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "stage1"


def shard_items(items: list, spec: str) -> list:
    i, n = (int(x) for x in spec.split("/"))
    return [x for k, x in enumerate(items) if k % n == i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--iters-a", type=int, default=250)
    ap.add_argument("--iters-b", type=int, default=700)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=1,
                    help="мультистарт: прогоны identify с seed 0..N-1, "
                         "оставляется лучший по V RMSE")
    args = ap.parse_args()

    quality = json.loads((ROOT / "configs" / "cell_quality.json").read_text())
    keep = [cid for cid, v in sorted(quality.items())
            if v.get("status") in ("ok", "early_failure")]
    cells = shard_items(keep, args.shard)
    if args.limit:
        cells = cells[: args.limit]
    print(f"шард {args.shard}: элементов {len(cells)} из {len(keep)}")

    cells_idx = {c.cell_id: c for c in list_cells(ROOT / "data" / "raw" / "aurora")}
    ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
    anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")
    CKPT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    log_path = ROOT / "reports" / f"stage1_all.{args.shard.replace('/', '_')}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    for k, cid in enumerate(cells):
        t_all = time.time()
        try:
            cf = cells_idx[cid]
            meta = parse_metadata(cf.metadata_path)
            df = load_bdf(cf.parquet_path)
            rep, cycles = build_cell_report(cid, df, meta)
            proc_path = PROC / f"{cid}.parquet"
            if proc_path.exists():
                pdf = pl.read_parquet(proc_path)
            else:
                proc = preprocess_cell(cid, df, cycles, rep.formation_cycles,
                                       set(rep.artefact_cycles), rep.current_sign)
                proc.df.write_parquet(proc_path)
                pdf = proc.df
            area_m2 = (meta.electrode_areas.get("positive_cm2") or 1.539) * 1e-4
            pseudo = build_pseudo_ocv(df, rep.formation_cycles)
            res = None
            for s_ in range(args.restarts):
                res_s = identify_cell(
                    cid, pdf, cycles, ocv_n, ocv_p, anchors,
                    formation_cycles=rep.formation_cycles,
                    area_m2=area_m2, stride=args.stride,
                    iters_a=args.iters_a, iters_b=args.iters_b,
                    v_min=rep.protocol_summary.get("v_min_main") or 2.5,
                    v_max=rep.protocol_summary.get("v_max_main") or 4.2,
                    pseudo=pseudo, verbose=False, seed=s_)
                if res is None or res_s.v_rmse_mv < res.v_rmse_mv:
                    res = res_s
            if args.refine:
                res = refine_per_cycle(res, pdf, cycles, ocv_n, ocv_p,
                                       area_m2=area_m2, iters=args.refine)
            import subprocess
            try:
                code_version = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                    text=True).strip()
            except Exception:
                code_version = "unknown"
            out = {
                "cell_id": res.cell_id, "anchor": res.anchor_name,
                "code_version": code_version,
                "restarts": args.restarts,
                "rho": res.rho, "q_n_ah": res.q_n_ah,
                "c_n": res.c_n, "c_p": res.c_p, "j0_mult": list(res.j0_mult),
                "ocv_p_scale": res.ocv_p_scale, "ocv_p_shift_v": res.ocv_p_shift_v,
                "ocv_n_scale": res.ocv_n_scale, "ocv_n_shift_v": res.ocv_n_shift_v,
                "cycles": res.cycles, "theta_n0": res.theta_n0,
                "theta_p0": res.theta_p0, "r_total_ohm": res.r_total_ohm,
                "q_li_ah": res.q_li_ah, "hyst_mv": res.hyst_mv,
                "lam_n": res.lam_n, "lam_p": res.lam_p,
                "ocv_p_delta": res.ocv_p_delta, "ocv_n_delta": res.ocv_n_delta,
                "v_rmse_mv": res.v_rmse_mv, "runtime_s": res.runtime_s,
                "j0n_per_cycle": getattr(res, "j0n_per_cycle", []),
                "j0p_per_cycle": getattr(res, "j0p_per_cycle", []),
            }
            (CKPT / f"{cid}.json").write_text(json.dumps(out, indent=1))
            # траектории для обучения деградации: (t, i, φ_n, θ_n, θ_p)
            t, ii, _, _ = pack_cycles(pdf, res.cycles)
            dt = res.debug_terms
            phi_n = (dt["u_n"] - dt["eta_n"]).numpy()
            np.savez_compressed(
                CKPT / f"{cid}.npz",
                t_s=t.numpy(), i_a=ii.numpy(), phi_n=phi_n,
                theta_n=dt["theta_n"].numpy(), theta_p=dt["theta_p"].numpy(),
                v_hat=res.v_hat.numpy(),
            )
            log(f"[{k+1}/{len(cells)}] {cid}: RMSE {res.v_rmse_mv:.1f} мВ, "
                f"якорь {res.anchor_name}, {res.runtime_s:.0f} с "
                f"(всего {time.time()-t_all:.0f} с)")
        except Exception:
            log(f"[{k+1}/{len(cells)}] {cid}: СБОЙ\n{traceback.format_exc(limit=3)}")


if __name__ == "__main__":
    main()
