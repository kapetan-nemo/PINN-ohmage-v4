#!/usr/bin/env python3
"""Мультистарт-диагностика ширины оврага минимумов идентификации.

N прогонов ``identify_cell`` с разными сидами (джиттер начальных точек)
на полной жизни элемента. Разброс итоговых траекторий каналов при
сопоставимой невязке — мера практической неидентифицируемости
(идентификационная неопределённость), кандидатная ось ансамбля.

Использование:
    uv run python scripts/diag_multistart.py <cell_id> [n_starts]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

from pinn_soh.data.bdf_loader import list_cells, load_bdf  # noqa: E402
from pinn_soh.data.metadata import parse_metadata  # noqa: E402
from pinn_soh.data.quality import build_cell_report  # noqa: E402
from pinn_soh.data.preprocess import build_pseudo_ocv  # noqa: E402
from pinn_soh.train.stage1_extract_state import (  # noqa: E402
    identify_cell, load_anchors, load_pretrained_ocv, refine_per_cycle,
)

cid = sys.argv[1]
n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 3

cf = {c.cell_id: c for c in list_cells(ROOT / "data" / "raw" / "aurora")}[cid]
meta = parse_metadata(cf.metadata_path)
df = load_bdf(cf.parquet_path)
rep, cycles = build_cell_report(cid, df, meta)
pdf = pl.read_parquet(ROOT / "data" / "processed" / f"{cid}.parquet")
area_m2 = (meta.electrode_areas.get("positive_cm2") or 1.539) * 1e-4
v_max = rep.protocol_summary.get("v_max_main") or 4.2
v_min = rep.protocol_summary.get("v_min_main") or 2.5
pseudo = build_pseudo_ocv(df, rep.formation_cycles)
ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")

runs = []
for s in range(n_starts):
    res = identify_cell(
        cid, pdf, cycles, ocv_n, ocv_p, anchors,
        formation_cycles=rep.formation_cycles, area_m2=area_m2,
        stride=20, iters_a=200, iters_b=500,
        v_min=v_min, v_max=v_max, pseudo=pseudo, verbose=False, seed=s)
    res = refine_per_cycle(res, pdf, cycles, ocv_n, ocv_p,
                           area_m2=area_m2, iters=200)
    runs.append(res)
    print(f"seed {s}: V RMSE {res.v_rmse_mv:.1f} мВ, "
          f"q_n {res.q_n_ah:.4f} Ah, rho {res.rho:.3f}, "
          f"{res.runtime_s/60:.1f} мин", flush=True)

# --- ширина оврага: разброс каналов по сидам ---------------------------
CHAN = ["theta_n0", "theta_p0", "r_total_ohm", "q_li_ah", "lam_n", "lam_p"]
print("\n=== ширина оврага (p2.5–p97.5 разброса по сидам) ===")
for ch in CHAN:
    arr = np.array([getattr(r, ch) for r in runs])
    if arr.ndim < 2 or arr.shape[1] == 0:
        continue
    half = (np.percentile(arr, 97.5, axis=0)
            - np.percentile(arr, 2.5, axis=0)) / 2
    med = np.median(arr, axis=0)
    rel = np.nanmedian(half / np.abs(med).clip(min=1e-6))
    print(f"  {ch:12s}: полуразмах p50 {np.median(half):.4f} "
          f"(относит. {rel:.1%}), max {half.max():.4f}")

spread = {ch: np.array([getattr(r, ch) for r in runs]).tolist()
          for ch in CHAN}
np.save(ROOT / "checkpoints" / f"multistart_{cid}.npy", spread,
        allow_pickle=True)
print(f"\nсохранено: checkpoints/multistart_{cid}.npy "
      f"({n_starts} прогонов, {len(runs[0].cycles)} окон)")
