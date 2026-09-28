"""Этап 1 (пилот): идентификация состояния на нескольких элементах.

Выбирает по одному элементу из каждой группы верхней отсечки напряжения
(из обучаемых, статус ok), выполняет предобработку и идентификацию,
печатает V RMSE, время счёта и начало траектории запаса лития.
Результаты — ``checkpoints/stage1/<cell>.json``.
"""

import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.data.bdf_loader import list_cells, load_bdf  # noqa: E402
from pinn_soh.data.metadata import parse_metadata  # noqa: E402
from pinn_soh.data.preprocess import build_pseudo_ocv, preprocess_cell  # noqa: E402
from pinn_soh.data.quality import build_cell_report  # noqa: E402
from pinn_soh.train.stage1_extract_state import (  # noqa: E402
    identify_cell, load_anchors, load_pretrained_ocv, refine_per_cycle,
)

PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "stage1"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None,
                    help="id элементов; без аргумента — по одному на группу V_max")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--iters-a", type=int, default=250)
    ap.add_argument("--iters-b", type=int, default=400)
    ap.add_argument("--refine", type=int, default=0,
                    help="итераций независимого поциклового уточнения (0 — выкл.)")
    args = ap.parse_args()

    cells_idx = {c.cell_id: c for c in list_cells(ROOT / "data" / "raw" / "aurora")}
    quality = json.loads((ROOT / "configs" / "cell_quality.json").read_text())
    eda = pl.read_parquet(ROOT / "reports" / "eda_cells.parquet")

    if args.cells:
        chosen = args.cells
    else:
        chosen = []
        seen_v = set()
        for row in eda.filter(pl.col("status") == "ok").sort("cell_id").iter_rows(named=True):
            key = row["v_max_main"]
            if key not in seen_v:
                seen_v.add(key)
                chosen.append(row["cell_id"])
    print("элементы для идентификации:", chosen)

    ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
    anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")

    CKPT.mkdir(parents=True, exist_ok=True)
    (PROC).mkdir(parents=True, exist_ok=True)
    for cid in chosen:
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
            proc.pseudo_ocv.write_parquet(PROC / f"{cid}.pseudo_ocv.parquet")
            pdf = proc.df
        area_cm2 = meta.electrode_areas.get("positive_cm2")
        area_m2 = (area_cm2 or 1.539) * 1e-4
        v_max = rep.protocol_summary.get("v_max_main") or 4.2
        v_min = rep.protocol_summary.get("v_min_main") or 2.5
        pseudo = build_pseudo_ocv(df, rep.formation_cycles)
        t0 = time.time()
        res = identify_cell(
            cid, pdf, cycles, ocv_n, ocv_p, anchors,
            formation_cycles=rep.formation_cycles,
            area_m2=area_m2, stride=args.stride,
            iters_a=args.iters_a, iters_b=args.iters_b,
            v_min=v_min, v_max=v_max, verbose=True,
            pseudo=pseudo,
        )
        if args.refine:
            res = refine_per_cycle(res, pdf, cycles, ocv_n, ocv_p,
                                   area_m2=area_m2, iters=args.refine,
                                   verbose=True)
        out = {
            "cell_id": res.cell_id, "rho": res.rho, "q_n_ah": res.q_n_ah,
            "c_n": res.c_n, "c_p": res.c_p, "j0_mult": list(res.j0_mult),
            "cycles": res.cycles, "theta_n0": res.theta_n0,
            "theta_p0": res.theta_p0, "r_total_ohm": res.r_total_ohm,
            "q_li_ah": res.q_li_ah, "v_rmse_mv": res.v_rmse_mv,
            "runtime_s": res.runtime_s,
            "lam_n": getattr(res, "lam_n", []), "lam_p": getattr(res, "lam_p", []),
            "hyst_mv": res.hyst_mv,
            "j0n_per_cycle": getattr(res, "j0n_per_cycle", []),
            "j0p_per_cycle": getattr(res, "j0p_per_cycle", []),
        }
        (CKPT / f"{cid}.json").write_text(json.dumps(out, indent=1))
        print(
            f"{cid}: V RMSE {res.v_rmse_mv:.1f} мВ | ρ={res.rho:.3f} "
            f"Q_n={res.q_n_ah * 1000:.3f} мА·ч | время {res.runtime_s:.0f} с"
        )
        print("   q_li начало/конец:", round(res.q_li_ah[0] * 1000, 4),
              "→", round(res.q_li_ah[-1] * 1000, 4), "мА·ч;",
              "θ_n0:", [round(x, 3) for x in res.theta_n0[:5]], "…")


if __name__ == "__main__":
    main()
