"""Аудит нефизичности прогноза: сравнение предсказанных внутренних
траекторий (q_li, R, delta_sei, SOH) с идентифицированными
полножизненными кривыми из checkpoints/stage1/*.json.

Для каждого тестового элемента рисуем 4 панели: SOH, q_li, R, delta.
Точки надлома прогноза (резкий спад/линейные участки) помечаются.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "reports" / "forecast" / (sys.argv[1] if len(sys.argv) > 1
                                       else "k100_kinC")
CKPT = ROOT / "checkpoints" / "stage1"
TEST = json.loads((ROOT / "configs" / "split.json").read_text())["test"]


def load_real(cid):
    f = CKPT / f"{cid}.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    return {k: np.asarray(m[k], float) for k in
            ("cycles", "q_li_ah", "r_total_ohm", "lam_n", "lam_p")
            if m.get(k) is not None}


rows = []
for cid in TEST:
    npz = RUN / f"{cid}_K100.npz"
    if not npz.exists():
        continue
    d = np.load(npz, allow_pickle=True)
    real = load_real(cid)
    rows.append((cid, d, real))

n = len(rows)
ncols = 4
fig, axes = plt.subplots(n, ncols, figsize=(16, 2.6 * n),
                         squeeze=False)
knee_log = []
for r_i, (cid, d, real) in enumerate(rows):
    cp, sp = d["cycles_pred"], d["soh_pred"]
    ct, st = d["cycles_true"], d["soh_true"]
    qp, rp, dp = d["q_li"], d["r_total"], d["delta_sei"]
    qdc = d["q_dch"] if "q_dch" in d.files else None

    ax = axes[r_i]
    ax[0].plot(ct, st, "k-", lw=1.2)
    ax[0].plot(cp, sp, "C0-", lw=1.2)
    ax[0].set_title(f"{cid[9:]} SOH", fontsize=8)
    ax[0].set_ylim(-0.05, 1.05)

    if real is not None:
        ax[1].plot(real["cycles"], real["q_li_ah"] * 1e3,
                   "k-", lw=1.2)
        ax[2].semilogy(real["cycles"], real["r_total_ohm"],
                       "k-", lw=1.2)
    ax[1].plot(cp, qp * 1e3, "C0-", lw=1.2)
    ax[1].set_title("q_li mAh", fontsize=8)
    ax[2].semilogy(cp, rp, "C0-", lw=1.2)
    ax[2].axhline(1e3, color="r", ls=":", lw=0.8)
    ax[2].set_title("R Ом", fontsize=8)
    ax[3].plot(cp, dp * 1e9, "C0-", lw=1.2)
    ax[3].set_title("δ_SEI нм", fontsize=8)

    # поиск резких просадок прогноза: |Δsoh/Δk| > 0.02/цикл
    dsp = np.diff(sp) / np.maximum(np.diff(cp), 1)
    jumps = np.where(dsp < -0.02)[0]
    for j in jumps:
        for a in ax:
            a.axvline(cp[j + 1], color="r", alpha=0.4, lw=0.7)
    if len(jumps):
        knee_log.append((cid, [int(cp[j + 1]) for j in jumps][:5]))
    # линейные участки: длинна >=8 шагов с почти постоянным наклоном
    for a in ax:
        a.tick_params(labelsize=7)
fig.suptitle(RUN.name + ": прогноз (синий) vs факт/идентиф. (чёрный)",
             fontsize=11)
fig.tight_layout()
out = RUN / "audit_states.png"
fig.savefig(out, dpi=110)
print("сохранено:", out)
print("\nРезкие просадки прогноза:")
for cid, js in knee_log:
    print(f"  {cid}: циклы {js}")

# сводка смещений по каналам на конце горизонта
print("\nСмещения конца горизонта (прогноз/факт):")
print(f"{'cell':22s} {'q_li':>7s} {'R':>7s} {'SOH':>7s}")
for cid, d, real in rows:
    if real is None:
        continue
    rp = np.interp(d["cycles_pred"], real["cycles"],
                   real["r_total_ohm"])
    qp_ = np.interp(d["cycles_pred"], real["cycles"],
                    real["q_li_ah"])
    sp_t = d["soh_true"]
    i_end = -1
    print(f"{cid:22s} {d['q_li'][i_end]/max(qp_[i_end],1e-9):7.2f} "
          f"{d['r_total'][i_end]/max(rp[i_end],1e-9):7.2f} "
          f"{d['soh_pred'][i_end]/max(sp_t[i_end],1e-9):7.2f}")
