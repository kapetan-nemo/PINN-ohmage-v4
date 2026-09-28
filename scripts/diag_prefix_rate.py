"""Диагностика темпа на стыке префикса (H3): пошаговые отношения
наблюдаемых приращений к предсказанным переходной моделью.

    .venv/bin/python scripts/diag_prefix_rate.py empa__ccid000051 100
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.data.bdf_loader import list_cells, load_bdf  # noqa: E402
from pinn_soh.data.metadata import parse_metadata  # noqa: E402
from pinn_soh.data.preprocess import build_pseudo_ocv  # noqa: E402
from pinn_soh.data.quality import build_cell_report  # noqa: E402
from pinn_soh.models.encoder import build_features  # noqa: E402
from pinn_soh.models.transition import TransitionNet  # noqa: E402
from pinn_soh.physics.degradation import DegradationConsts  # noqa: E402
from pinn_soh.physics.cell import F_CONST, R_CONST, T_REF  # noqa: E402
from pinn_soh.predict.forecast import fit_z_prefix  # noqa: E402
from pinn_soh.train.stage1_extract_state import (  # noqa: E402
    identify_cell, load_anchors, load_pretrained_ocv, pack_cycles,
    refine_per_cycle,
)

cid, K = sys.argv[1], int(sys.argv[2])
cf = {c.cell_id: c for c in list_cells(ROOT / "data" / "raw" / "aurora")}[cid]
meta = parse_metadata(cf.metadata_path)
df = load_bdf(cf.parquet_path)
rep, cycles = build_cell_report(cid, df, meta)
pdf = pl.read_parquet(ROOT / "data" / "processed" / f"{cid}.parquet")
area_m2 = (meta.electrode_areas.get("positive_cm2") or 1.539) * 1e-4
v_max = rep.protocol_summary.get("v_max_main") or 4.2
v_min = rep.protocol_summary.get("v_min_main") or 2.5
first_main = rep.formation_cycles + 1
pseudo = build_pseudo_ocv(df, rep.formation_cycles)
ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")
consts = DegradationConsts.from_json(ROOT / "data" / "params" / "kinetics_OKane2022.json")
z_pop = torch.tensor(
    json.loads((ROOT / "data" / "params" / "z_population.json").read_text())["median"],
    dtype=torch.float64)
trans = TransitionNet().double()
trans.load_state_dict(
    torch.load(ROOT / "checkpoints" / "stage3" / "transition_local.pt",
               weights_only=False)["transition"], strict=False)
trans.eval()

res = identify_cell(cid, pdf, cycles, ocv_n, ocv_p, anchors,
                    formation_cycles=rep.formation_cycles, area_m2=area_m2,
                    stride=20, iters_a=200, iters_b=500,
                    v_min=v_min, v_max=v_max, pseudo=pseudo,
                    max_cycle=first_main + K - 1, verbose=False)
res = refine_per_cycle(res, pdf, cycles, ocv_n, ocv_p, area_m2=area_m2, iters=200)
md = dict(res.__dict__)
ac_ = None
for nm, a in anchors.get("positive", {}).items():
    if nm == res.anchor_name:
        ac_ = torch.tensor(a["code"])
feats = build_features(md, ac_)
z = fit_z_prefix(res, pdf, consts, iters=150, z0=z_pop)

# --- реплика _prefix_gains: пошаговые obs/pred ---------------------------
k_sei = consts.k_sei_m_s * (10.0 ** float(z[0]))
d_solv = consts.d_solv_m2_s * (10.0 ** float(z[1]))
j0_pl = consts.j0_pl_a_m2 * (10.0 ** float(z[2]))
rho_sei = consts.rho_sei_ohm_m * (10.0 ** float(z[3]))
area_eff = 10.0 ** float(z[4])
a_sei = consts.alpha_sei * F_CONST / (R_CONST * T_REF)
a_pl = consts.alpha_pl * F_CONST / (R_CONST * T_REF)
qli0_ = res.q_li_ah[0]

tt_, ii_, _, _ = pack_cycles(pdf, res.cycles)
dt = res.debug_terms
phi = (dt["u_n"] - dt["eta_n"]).double()
w_ = torch.zeros(phi.shape, dtype=torch.float64)
dts_ = tt_[:, 1:] - tt_[:, :-1]
w_[:, 0], w_[:, -1] = 0.5 * dts_[:, 0], 0.5 * dts_[:, -1]
w_[:, 1:-1] = 0.5 * (dts_[:, :-1] + dts_[:, 1:])
expo_ = torch.exp(a_sei * (phi - consts.u_sei_v)).clamp(max=1e6)
expp_ = torch.exp((a_pl * phi).clamp(max=50.0))
expm_ = torch.exp((-a_pl * phi).clamp(max=50.0))
j_pl_ = torch.where(phi < 0.0, j0_pl * (expp_ - expm_), torch.zeros_like(phi))
jpl = (j_pl_.abs() * w_).sum(dim=1)

lam_np = np.asarray(res.lam_n or [1.0] * len(res.cycles))
lam_pp = np.asarray(res.lam_p or [1.0] * len(res.cycles))
q_arr = np.asarray(res.q_li_ah)
r_arr = np.asarray(res.r_total_ohm)
cyc = np.asarray(res.cycles, float)
d_ = consts.delta_sei0_m
g0 = max(cyc[1] - cyc[0], 1.0)
dq_p = max(q_arr[0] - q_arr[1], 0.0) / g0 / qli0_
rows = []
for m in range(len(cyc) - 1):
    gap = cyc[m + 1] - cyc[m]
    denom = d_ / d_solv + expo_[m] / k_sei
    j_sei_ = (-F_CONST * consts.c_solv_mol_m3 / denom).abs()
    int_sei_ = float((j_sei_ * w_[m]).sum()) * area_eff / 3600.0
    int_pl_ = float(jpl[m]) * area_eff / 3600.0
    dqm = int_sei_ + consts.beta_dead * int_pl_
    d_ += int_sei_ * 3600.0 * consts.v_sei_m3_mol / (F_CONST * area_eff)
    dyn_ = torch.tensor(
        [[q_arr[m] / qli0_, lam_np[m], lam_pp[m],
          math.log10(max(r_arr[m] / res.r_total_ohm[0], 1e-6)),
          dqm / qli0_, math.log10(cyc[m] + 1.0) / 4.0,
          gap / 20.0, dq_p]], dtype=torch.float64)
    with torch.no_grad():
        o_ = trans(dyn_, feats.reshape(1, -1))[0]
    dq_p = float(o_[0]) * dqm / qli0_ + float(o_[1])
    obs_q = (q_arr[m] - q_arr[m + 1]) / qli0_
    rows.append((cyc[m], cyc[m + 1], obs_q * gap / gap,
                 dq_p * gap, obs_q / (dq_p) if dq_p > 1e-9 else np.nan))

print(f"\n{cid}, K={K}: идентифицировано {len(cyc)} циклов")
print("  цикл_a→b     obs Δq/q0     pred Δq/q0     ratio")
for a, b, o, p, rt in rows:
    print(f"  {a:4.0f}→{b:4.0f}   {o:11.5f}   {p:12.5f}   {rt:8.2f}")
rat = np.asarray([r[4] for r in rows])
fin = np.isfinite(rat)
print("\nмедиана ratio: все %.2f | поздняя половина %.2f | последние 4 %.2f"
      % (np.nanmedian(rat[fin]),
         np.nanmedian(rat[fin][len(rat[fin]) // 2:]),
         np.nanmedian(rat[fin][-4:])))
# темп наблюдаемых приращений: ранняя/поздняя половина и хвост
obs = np.asarray([r[2] for r in rows])
n = len(obs)
print("obs Δq/q0: ранняя пол. %.5f | поздняя %.5f | посл.4 %.5f"
      % (np.median(obs[:n // 2]), np.median(obs[n // 2:]), np.median(obs[-4:])))
print("наклон obs (посл.половина):",
      np.polyfit(np.arange(len(obs[n // 2:])), obs[n // 2:], 1)[0])
