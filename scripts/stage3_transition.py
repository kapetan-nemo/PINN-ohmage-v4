"""Этап 3: обучение переходной модели состояния (TransitionNet).

По траекториям этапа 1 (q_li, λ_n, λ_p, R на идентифицированных циклах)
учит приращения за цикл: множитель механического Δq, остаток Δq,
Δλ_n, Δλ_p, ΔR. Потеря — **развёртка**: модель рекурсивно ведёт
состояние от первой идентифицированной точки до конца жизни, ошибка
считается по всей траектории (one-step обучение давало расходимость
свободного прогона). Плюс лёгкий one-step якорь.

    .venv/bin/python scripts/stage3_transition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.models.transition import TransitionNet  # noqa: E402
from pinn_soh.physics.degradation import DegradationConsts  # noqa: E402
from pinn_soh.train.stage2_degradation import (  # noqa: E402
    load_cell_data, mech_increments,
)

CKPT = ROOT / "checkpoints" / "stage1"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "checkpoints" / "stage3"


def pad_cells(cells, z_pop, consts):
    """Пакет (C, B) с маской: наблюдения состояния и входы модели."""
    cells = [c for c in cells if c.lam_n is not None and len(c.cycles) >= 6]
    C, B = len(cells), max(len(c.cycles) for c in cells)
    S = np.zeros((C, B, 4))            # наблюдения: q_rel, lam_n, lam_p, r_rel
    D = np.zeros((C, B - 1, 3))        # дин. входы: dqm_rate, log_k, gap
    M = np.zeros((C, B - 1))           # маска шагов
    F = np.zeros((C, cells[0].feats.shape[0]))
    for i, c in enumerate(cells):
        b = len(c.cycles)
        dq_mech, _ = mech_increments(c, z_pop, consts)
        gap = np.diff(c.cycles)
        q0, r0 = c.q_li[0], c.r_total[0]
        # фильтрация выбросов идентификации: точки вне физического
        # коридора исключаются из потери (состояние всё же эволюционирует)
        med_r = np.median(c.r_total)
        ok_r = (c.r_total > 0.3 * med_r) & (c.r_total < 3.0 * med_r)
        ok_l = (np.asarray(c.lam_n) > 0.25) & (np.asarray(c.lam_n) < 1.65) \
            & (np.asarray(c.lam_p) > 0.25) & (np.asarray(c.lam_p) < 1.65)
        ok_q = (c.q_li > 0.2 * q0) & (c.q_li < 1.3 * q0)
        ok = ok_r & ok_l & ok_q
        S[i, :b] = np.stack([c.q_li / q0, c.lam_n, c.lam_p,
                             np.log10(c.r_total / r0)], axis=1)
        D[i, :b - 1] = np.stack([
            dq_mech / gap / q0,
            np.log10(np.asarray(c.cycles[:-1]) + 1.0) / 4.0,
            gap / 20.0], axis=1)
        M[i, :b - 1] = ok[1:]
        F[i] = c.feats.numpy()
    return (torch.tensor(S), torch.tensor(D), torch.tensor(M),
            torch.tensor(F), cells)


def rollout(net, S, D, M, F):
    """Рекурсивная развёртка состояния от первой точки (дифференцируемо)."""
    C, B, _ = S.shape
    q, ln, lp, rr = S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 0, 3]
    # инерция: первый шаг получает наблюдаемую скорость, далее —
    # предсказанную моделью (скрытое состояние «режима»)
    gap0 = (D[:, 0, 2] * 20.0).clamp(min=1.0)
    dq_prev = ((S[:, 0, 0] - S[:, 1, 0]) / gap0).clamp(min=0.0)
    pred = [torch.stack([q, ln, lp, rr], 1)]
    for k in range(B - 1):
        dyn = torch.stack([q, ln, lp, rr, D[:, k, 0], D[:, k, 1],
                           D[:, k, 2], dq_prev], dim=1)
        o = net(dyn, F)
        gap = D[:, k, 2] * 20.0
        dq_prev = o[:, 0] * D[:, k, 0] + o[:, 1]
        q = (q - gap * dq_prev).clamp(min=1e-3)
        ln = (ln - gap * o[:, 2]).clamp(0.3, 1.6)
        lp = (lp - gap * o[:, 3]).clamp(0.3, 1.6)
        rr = rr + gap * o[:, 4]
        pred.append(torch.stack([q, ln, lp, rr], 1))
    return torch.stack(pred, 1)          # (C, B, 4)


def huber(d: torch.Tensor, eps: float) -> torch.Tensor:
    a = d.abs()
    return torch.where(a <= eps, 0.5 * d ** 2, eps * (a - 0.5 * eps))


def local_loss(net, S, D, M, F, eps):
    """Одношаговая потеря: (наблюдаемое s_k, входы) → Δs_k.

    Развёртка «в среднем» объясняет середину жизни постоянным наклоном
    и не учит замедление; прямое сопоставление приращений по каждому
    наблюдаемому шагу заставляет сеть использовать log_k.
    """
    C, B, _ = S.shape
    tot, cnt = 0.0, 0.0
    for k in range(B - 1):
        gap = (D[:, k, 2] * 20.0).clamp(min=1.0)
        if k == 0:
            dq_prev = ((S[:, 0, 0] - S[:, 1, 0]) / gap).clamp(min=0.0)
        else:
            g0 = (D[:, k - 1, 2] * 20.0).clamp(min=1.0)
            dq_prev = ((S[:, k - 1, 0] - S[:, k, 0]) / g0).clamp(min=0.0)
        dyn = torch.stack([S[:, k, 0], S[:, k, 1], S[:, k, 2], S[:, k, 3],
                           D[:, k, 0], D[:, k, 1], D[:, k, 2], dq_prev], 1)
        o = net(dyn, F)
        pred = torch.stack([
            o[:, 0] * D[:, k, 0] + o[:, 1],     # Δq за цикл
            o[:, 2], o[:, 3], -o[:, 4]], dim=1)  # Δλ, −Δlog r (знак!)
        obs = torch.stack([
            (S[:, k, 0] - S[:, k + 1, 0]) / gap,
            (S[:, k, 1] - S[:, k + 1, 1]) / gap,
            (S[:, k, 2] - S[:, k + 1, 2]) / gap,
            (S[:, k, 3] - S[:, k + 1, 3]) / gap], dim=1)
        d = huber(pred - obs, eps / 20.0) * M[:, k].unsqueeze(-1)
        tot = tot + d.sum(); cnt = cnt + M[:, k].sum()
    return tot / cnt.clamp(min=1)


def main() -> None:
    split = json.loads((ROOT / "configs" / "split.json").read_text())
    quality = json.loads((ROOT / "configs" / "cell_quality.json").read_text())
    anchors = json.loads(
        (ROOT / "data" / "params" / "latent_anchors.json").read_text())
    kin = ROOT / "data" / "params" / "kinetics_OKane2022.json"
    consts = DegradationConsts.from_json(kin) if kin.exists() \
        else DegradationConsts()
    zp = ROOT / "data" / "params" / "z_population.json"
    z_pop = torch.tensor(json.loads(zp.read_text())["median"],
                         dtype=torch.float64) if zp.exists() \
        else torch.zeros(5, dtype=torch.float64)

    def load(ids):
        return [c for c in
                (load_cell_data(i, CKPT, PROC, quality, anchors)
                 for i in ids) if c is not None]

    cells_tr, cells_te = load(split["train"]), load(split["val"])
    S_tr, D_tr, M_tr, F_tr, cells_tr = pad_cells(cells_tr, z_pop, consts)
    S_te, D_te, M_te, F_te, cells_te = pad_cells(cells_te, z_pop, consts)
    S_ts, D_ts, M_ts, F_ts, cells_ts = pad_cells(
        load(split["test"]), z_pop, consts)
    print(f"train {len(cells_tr)} ({S_tr.shape[1]} шагов), "
          f"val {len(cells_te)}, test {len(cells_ts)}", flush=True)

    net = TransitionNet().double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    eps = torch.tensor([0.01, 0.01, 0.01, 0.02], dtype=torch.float64)
    Mm_tr = torch.cat([M_tr, torch.ones(len(M_tr), 1).double()], 1) \
        .unsqueeze(-1)
    Mm_te = torch.cat([M_te, torch.ones(len(M_te), 1).double()], 1) \
        .unsqueeze(-1)
    Mm_ts = torch.cat([M_ts, torch.ones(len(M_ts), 1).double()], 1) \
        .unsqueeze(-1)
    best = (float("inf"), None)
    for ep in range(1500):
        opt.zero_grad()
        P = rollout(net, S_tr, D_tr, M_tr, F_tr)
        d = huber(P - S_tr, eps) * Mm_tr
        loss = d.sum() / Mm_tr.sum().clamp(min=1)
        loss = loss + 2.0 * local_loss(net, S_tr, D_tr, M_tr, F_tr, eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        # отбор по val каждые 25 эпох (дешёвый прогон)
        if ep % 25 == 0 or ep == 1499:
            with torch.no_grad():
                Pe = rollout(net, S_te, D_te, M_te, F_te)
                de = huber(Pe - S_te, eps) * Mm_te
                te = float(de.sum() / Mm_te.sum().clamp(min=1))
            if te < best[0]:
                best = (te, {k: v.clone() for k, v in
                             net.state_dict().items()})
        if ep % 150 == 0 or ep == 1499:
            with torch.no_grad():
                se = ((Pe - S_te) ** 2 * Mm_te).sum((0, 1)) \
                    / Mm_te.sum().clamp(min=1)
            print(f"ep {ep:4d}: train {float(loss.detach()):.2e}  "
                  f"val {te:.2e}  "
                  f"RMSE q {float(se[0])**.5:.4f} "
                  f"λn {float(se[1])**.5:.4f} "
                  f"λp {float(se[2])**.5:.4f} "
                  f"r {float(se[3])**.5:.4f}", flush=True)

    if best[1] is not None:
        net.load_state_dict(best[1])
        print(f"выбран чекпоинт по val: loss {best[0]:.2e}")

    # финальная оценка на отложенном test
    with torch.no_grad():
        Ps = rollout(net, S_ts, D_ts, M_ts, F_ts)
        ss = ((Ps - S_ts) ** 2 * Mm_ts).sum((0, 1)) \
            / Mm_ts.sum().clamp(min=1)
        print(f"TEST (n={len(cells_ts)}): RMSE q {float(ss[0])**.5:.4f} "
              f"λn {float(ss[1])**.5:.4f} λp {float(ss[2])**.5:.4f} "
              f"log-r {float(ss[3])**.5:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"transition": net.state_dict()}, OUT / "transition_local.pt")
    print("сохранено:", OUT / "transition_local.pt")


if __name__ == "__main__":
    main()
