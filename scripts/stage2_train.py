"""Обучение и оценка модели деградации (этап 2).

Загружает контрольные точки этапа 1 по сплиту ``configs/split.json``,
обучает кодирующую сеть, сравнивает с базовыми моделями A/B/C по
метрикам SOH на горизонтах и ошибкам пересечения порогов.

    .venv/bin/python scripts/stage2_train.py --epochs 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.baselines import (  # noqa: E402
    fit_empirical, predict_empirical, predict_sohnet, train_sohnet,
)
from pinn_soh.eval.metrics import soh_rmse_by_horizon, threshold_errors  # noqa: E402
from pinn_soh.models.encoder import FEATURE_DIM  # noqa: E402
from pinn_soh.physics.degradation import DegradationConsts  # noqa: E402
from pinn_soh.train.stage2_degradation import (  # noqa: E402
    CellData, load_cell_data, rollout_state, train_encoder,
)

CKPT = ROOT / "checkpoints" / "stage1"
PROC = ROOT / "data" / "processed"


def soh_true(cell: CellData) -> tuple[np.ndarray, np.ndarray]:
    """Измеренная нормированная разрядная ёмкость на идентифицированных циклах."""
    cyc, val = [], []
    for c in cell.cycles:
        q = (cell.q_dch_meas or {}).get(int(c))
        if q is not None:
            cyc.append(c)
            val.append(q)
    cyc, val = np.asarray(cyc), np.asarray(val)
    return cyc, val / val[0]


def eval_model(cells: list[CellData], predict_fn) -> dict:
    """Оценка произвольной функции прогноза cell → (cycles, soh_pred)."""
    rm_h, thr = {}, {}
    n_ok = 0
    for cell in cells:
        ct, st = soh_true(cell)
        if len(ct) < 5:
            continue
        cp, sp = predict_fn(cell)
        if sp is None:
            continue
        n_ok += 1
        for h, v in soh_rmse_by_horizon(cp, sp, ct, st).items():
            rm_h.setdefault(h, []).append(v)
        for lv, e in threshold_errors(cp, sp, ct, st).items():
            if e is not None:
                thr.setdefault(lv, []).append(e)
    return {
        "n_cells": n_ok,
        "soh_rmse": {h: float(np.mean(v)) for h, v in rm_h.items()},
        "threshold_mae_cycles": {lv: float(np.mean(np.abs(v)))
                                 for lv, v in thr.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    split = json.loads((ROOT / "configs" / "split.json").read_text())
    quality = json.loads((ROOT / "configs" / "cell_quality.json").read_text())
    anchors = json.loads((ROOT / "data" / "params" / "latent_anchors.json").read_text())
    kinetics = ROOT / "data" / "params" / "kinetics_OKane2022.json"
    consts = DegradationConsts.from_json(kinetics) if kinetics.exists() \
        else DegradationConsts()

    train_ids = [c for c in split["train"] if (CKPT / f"{c}.json").exists()]
    test_ids = [c for c in split.get("test", split.get("val", []))
                if (CKPT / f"{c}.json").exists()]
    print(f"элементов с состоянием: train {len(train_ids)}, test {len(test_ids)}")

    t0 = time.time()
    cells_tr = [d for d in (load_cell_data(c, CKPT, PROC, quality, anchors)
                            for c in train_ids) if d is not None]
    cells_te = [d for d in (load_cell_data(c, CKPT, PROC, quality, anchors)
                            for c in test_ids) if d is not None]
    print(f"загружено: train {len(cells_tr)}, test {len(cells_te)} "
          f"({time.time()-t0:.0f} с)")
    # отбрасываем элементы с слишком короткой траекторией
    cells_tr = [c for c in cells_tr if len(c.cycles) >= 5]
    cells_te = [c for c in cells_te if len(c.cycles) >= 5]

    # --- модель C: чисто механистическая (z = 0)
    res_c = eval_model(cells_te, lambda cell: (
        cell.cycles,
        (rollout_state(cell, torch.zeros(5, dtype=torch.float64), consts)
         ["q_li"].numpy() / cell.q_li[0])))
    print("C (z=0):", json.dumps(res_c, indent=1))

    # --- модель A: эмпирическая √-кривая по первым 5 точкам
    def pred_a(cell: CellData):
        ct, st = soh_true(cell)
        kf = min(5, len(ct) - 1)
        coef = fit_empirical(ct[:kf], st[:kf])
        return cell.cycles, predict_empirical(coef, cell.cycles, ct[0])
    res_a = eval_model(cells_te, pred_a)
    print("A (эмпирика):", json.dumps(res_a, indent=1))

    # --- модель B: SOHNet по всем циклам train
    recs = [{"feats": c.feats, "cycles": c.cycles,
             "soh": c.q_li / c.q_li[0], "k_max": c.cycles[-1]}
            for c in cells_tr]
    net = train_sohnet(recs, FEATURE_DIM, epochs=800)
    res_b = eval_model(cells_te, lambda cell: (
        cell.cycles,
        predict_sohnet(net, cell.feats, cell.cycles, cell.cycles[-1])))
    print("B (нейросеть):", json.dumps(res_b, indent=1))

    # --- механистическая модель с кодирующей сетью
    enc = train_encoder(cells_tr, consts, epochs=args.epochs, lr=args.lr)
    def pred_pinn(cell: CellData):
        with torch.no_grad():
            pr = rollout_state(cell, enc(cell.feats), consts)
        return cell.cycles, pr["q_li"].numpy() / cell.q_li[0]
    res_pinn = eval_model(cells_te, pred_pinn)
    print("PINN (encoder+кинетика):", json.dumps(res_pinn, indent=1))

    out = {"A_empirical": res_a, "B_sohnet": res_b,
           "C_mechanistic_z0": res_c, "PINN": res_pinn}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "stage2_metrics.json").write_text(
        json.dumps(out, indent=1))
    (ROOT / "checkpoints" / "stage2").mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict()}, ROOT / "checkpoints" / "stage2" / "encoder.pt")
    print("сохранено: reports/stage2_metrics.json, checkpoints/stage2/encoder.pt")


if __name__ == "__main__":
    main()
