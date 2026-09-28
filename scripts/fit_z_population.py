"""Популяционная оценка кинетических множителей z по элементам train-сплита.

Для каждого элемента с контрольной точкой этапа 1 подгоняет z на всей
идентифицированной траектории (уровень q_li + лог-R + приращения).
Результат: ``data/params/z_population.json`` — z по элементам,
медиана и MAD популяции. Медиана используется как априор в
``fit_z_prefix`` при честном прогнозе (вместо z=0, который завышает
скорость LLI на порядки).

    .venv/bin/python scripts/fit_z_population.py --shard 0 --nshards 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.physics.degradation import DegradationConsts  # noqa: E402
from pinn_soh.train.stage2_degradation import (  # noqa: E402
    load_cell_data, loss_increments, loss_rollout, rollout_state,
)

CKPT = ROOT / "checkpoints" / "stage1"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "params" / "z_population.json"


def fit_z_cell(cell, consts, iters: int = 400, lr: float = 0.08) -> torch.Tensor:
    """Поэлементная подгонка z на полной траектории (слабый априор к 0)."""
    z = torch.zeros(5, dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        pred = rollout_state(cell, z, consts)
        loss = loss_rollout(cell, pred) + loss_increments(cell, pred) \
            + 0.02 * (z ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_value_(z, 5.0)
        opt.step()
    return z.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    split = json.loads((ROOT / "configs" / "split.json").read_text())
    quality = json.loads((ROOT / "configs" / "cell_quality.json").read_text())
    anchors = json.loads(
        (ROOT / "data" / "params" / "latent_anchors.json").read_text())
    kin = ROOT / "data" / "params" / "kinetics_OKane2022.json"
    consts = DegradationConsts.from_json(kin) if kin.exists() \
        else DegradationConsts()

    cells = sorted(split["train"])
    shard = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]
    part = ROOT / "data" / "params" / f"z_population_part{args.shard}.json"
    res = json.loads(part.read_text()) if part.exists() else {}
    for n, cid in enumerate(shard):
        if cid in res:
            continue
        cell = load_cell_data(cid, CKPT, PROC, quality, anchors)
        if cell is None:
            continue
        try:
            z = fit_z_cell(cell, consts, iters=args.iters)
            res[cid] = [float(x) for x in z]
            if n % 10 == 0:
                print(f"[{args.shard}] {cid}: z={np.round(res[cid], 2)}",
                      flush=True)
        except Exception as e:
            print(f"[{args.shard}] {cid}: СБОЙ {e}", flush=True)
        part.write_text(json.dumps(res, indent=1))

    # слияние только когда собраны все части (одиночный запуск nshards=1)
    if args.nshards == 1:
        zs = np.asarray(list(res.values()))
        med, mad = np.median(zs, 0), np.median(np.abs(zs - np.median(zs, 0)), 0)
        OUT.write_text(json.dumps({
            "z_per_cell": res,
            "median": med.tolist(), "mad": mad.tolist()}, indent=1))
        print("медиана z:", np.round(med, 2), "MAD:", np.round(mad, 2))


if __name__ == "__main__":
    main()
