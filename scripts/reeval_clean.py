"""Пересчёт метрик сохранённых прогонов на очищенной эталонной
кривой SOH (без усечённых циклов, duration < 0.7·медианы)."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed" / "cycles"
RUNS = sys.argv[1:] or ["k100_pers", "k100_kinC", "k100_kinE"]


def clean_truth(cid, ct, st):
    """Маска усечённых циклов из длительностей; ct/st — уже
    нормированная кривая из npz."""
    f = PROC / f"{cid}.parquet"
    if not f.exists():
        return None
    cy = pl.read_parquet(f).sort("cycle")
    ids = cy["cycle"].to_numpy()
    dur = cy["duration_s"].to_numpy() \
        if "duration_s" in cy.columns else np.ones(len(ids))
    med = np.median(dur[5:]) if len(dur) > 10 else np.median(dur)
    ok_ids = set(ids[dur >= 0.7 * med].tolist())
    ok = np.array([c in ok_ids for c in ct])
    return ct[ok], st[ok]


def threshold(c, s, level=0.8):
    m = s < level
    return float(c[m.argmax()]) if m.any() else None


for run in RUNS:
    rdir = ROOT / "reports" / "forecast" / run
    if not rdir.exists():
        continue
    rms, e80, n = [], [], 0
    for npz in sorted(rdir.glob("*_K100.npz")):
        cid = npz.name.split("_K")[0]
        d = np.load(npz, allow_pickle=True)
        tr = clean_truth(cid, d["cycles_true"], d["soh_true"])
        if tr is None:
            continue
        ct, st = tr
        st_i = np.interp(d["cycles_pred"], ct, st)
        rms.append(float(np.sqrt(np.mean((d["soh_pred"] - st_i) ** 2))))
        k80p = threshold(d["cycles_pred"], d["soh_pred"])
        k80t = threshold(ct, st)
        if k80p and k80t:
            e80.append(abs(k80p - k80t))
        n += 1
    if n:
        print(f"{run:16s} n={n:2d}  RMSE mean={np.mean(rms):.3f} "
              f"median={np.median(rms):.3f}  k80MAE={np.mean(e80):.0f} "
              f"(n={len(e80)})")
