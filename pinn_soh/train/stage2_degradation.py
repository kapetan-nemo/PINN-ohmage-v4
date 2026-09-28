"""Этап 2: обучение модели деградации по извлечённым состояниям.

Вход — контрольные точки этапа 1: для каждого элемента траектория
``q_li[k], r_total[k]`` по идентифицированным циклам и траектории
``φ_n(t)`` этих циклов. Рекурсивный прогон состояния::

    s_{k+1} = s_k + Δ(s_k; φ̂_n(t) цикла k, z)

где приращения — интегралы кинетики SEI и осаждения из
``physics/degradation.py``, ``z = encoder(признаки)``. Цель — воспроизвести
измеренную/идентифицированную траекторию запаса лития и сопротивления.

Интеграл за один идентифицированный цикл умножается на шаг до следующего
(приближение медленной динамики внутри шага).

Интерфейсы:
    ``load_cell_data`` — загрузка данных элемента;
    ``rollout_state`` — рекурсивный прогон медленного состояния;
    ``train_encoder`` — обучение кодирующей сети на train-сплите.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from pinn_soh.models.encoder import CellEncoder, build_features
from pinn_soh.physics.degradation import DegradationConsts, F_CONST, T_REF


@dataclass
class CellData:
    """Данные элемента для обучения деградации."""

    cell_id: str
    cycles: np.ndarray            # идентифицированные циклы (B)
    q_li: np.ndarray              # запас лития, А·ч (B)
    r_total: np.ndarray           # полное сопротивление, Ом (B)
    t_s: torch.Tensor             # (B, N)
    i_a: torch.Tensor             # (B, N)
    phi_n: torch.Tensor           # потенциал анода vs Li, В (B, N)
    q_dch_meas: np.ndarray | None  # измеренная разрядная ёмкость по циклам
    feats: torch.Tensor           # (F,)
    lam_n: np.ndarray | None = None  # относит. ёмкость анода по циклам (B)
    lam_p: np.ndarray | None = None  # относит. ёмкость катода (B)


def load_cell_data(cell_id: str, ckpt_dir: Path, proc_dir: Path,
                   quality: dict, anchors: dict) -> CellData | None:
    """Собирает данные элемента из контрольных точек этапа 1."""
    js, npz = ckpt_dir / f"{cell_id}.json", ckpt_dir / f"{cell_id}.npz"
    if not (js.exists() and npz.exists()):
        return None
    meta = json.loads(js.read_text())
    tr = np.load(npz)
    meta.update(quality.get(cell_id, {}))
    anchor_code = None
    for name, a in anchors.get("positive", {}).items():
        if name == meta["anchor"]:
            anchor_code = torch.tensor(a["code"])
    feats = build_features(meta, anchor_code)
    q_dch = None
    pf = proc_dir / f"{cell_id}.parquet"
    if pf.exists():
        df = pl.read_parquet(pf)
        dch = df.filter(pl.col("i_a") < -1e-4)
        q_dch = (dch.with_columns(
                    pl.col("t_s").diff().over("cycle").fill_null(0.0)
                    .alias("dt"))
                 .group_by("cycle")
                 .agg((-pl.col("i_a") * pl.col("dt")).sum().alias("q_ah") / 1.0)
                 .sort("cycle"))
        q_dch = dict(zip(q_dch["cycle"].to_numpy(),
                         (q_dch["q_ah"] / 3600.0).to_numpy()))
    return CellData(
        cell_id=cell_id,
        cycles=np.asarray(meta["cycles"], float),
        q_li=np.asarray(meta["q_li_ah"], float),
        r_total=np.asarray(meta["r_total_ohm"], float),
        t_s=torch.from_numpy(tr["t_s"]).double(),
        i_a=torch.from_numpy(tr["i_a"]).double(),
        phi_n=torch.from_numpy(tr["phi_n"]).double(),
        q_dch_meas=q_dch,
        feats=feats,
        lam_n=np.asarray(meta.get("lam_n"), float)
              if meta.get("lam_n") is not None else None,
        lam_p=np.asarray(meta.get("lam_p"), float)
              if meta.get("lam_p") is not None else None,
    )


def trapz_weights(t: torch.Tensor) -> torch.Tensor:
    """Веса трапеций для неравномерной сетки (B, N) → (B, N)."""
    n = t.shape[1]
    dt = t[:, 1:] - t[:, :-1]
    w = torch.zeros_like(t)
    w[:, 0] = 0.5 * dt[:, 0]
    w[:, -1] = 0.5 * dt[:, -1]
    if n > 2:
        w[:, 1:-1] = 0.5 * (dt[:, :-1] + dt[:, 1:])
    return w


def rollout_state(cell: CellData, z: torch.Tensor,
                  consts: DegradationConsts,
                  resid: torch.nn.Module | None = None,
                  temp_k: float = T_REF) -> dict:
    """Рекурсивный прогон медленного состояния по идентифицированным циклам.

    ``z`` — вектор множителей (Z_DIM); возвращает предсказанные
    ``q_li[k], r[k]``, ``delta[k]`` и приращения по шагам.
    """
    area_eff = 10.0 ** z[4]  # м²
    w = trapz_weights(cell.t_s)                       # (B, N)
    cycles = torch.tensor(cell.cycles, dtype=torch.float64)
    q_li_true = torch.tensor(cell.q_li, dtype=torch.float64)

    # эффективные константы — тензоры (сохраняется граф градиентов по z)
    d_solv = consts.d_solv_m2_s * (10.0 ** z[1])
    k_sei = consts.k_sei_m_s * (10.0 ** z[0])
    j0_pl = consts.j0_pl_a_m2 * (10.0 ** z[2])
    rho_sei = consts.rho_sei_ohm_m * (10.0 ** z[3])
    a_sei = consts.alpha_sei * F_CONST / (8.314462618 * temp_k)
    a_pl = consts.alpha_pl * F_CONST / (8.314462618 * temp_k)
    expo = torch.exp(a_sei * (cell.phi_n - consts.u_sei_v)).clamp(max=1e6)
    expp = torch.exp((a_pl * cell.phi_n).clamp(max=50.0))
    expm = torch.exp((-a_pl * cell.phi_n).clamp(max=50.0))
    j_pl = j0_pl * (expp - expm)
    j_pl = torch.where(cell.phi_n < 0.0, j_pl, torch.zeros_like(j_pl))
    jpl_int = (j_pl.abs() * w).sum(dim=1)             # (B,) А·с/м²

    delta = torch.tensor(consts.delta_sei0_m, dtype=torch.float64)
    r0 = float(cell.r_total[0])

    q_li_pred, r_pred, deltas = [q_li_true[0]], [cell.r_total[0]], [delta]
    for k in range(len(cell.cycles) - 1):
        gap = cycles[k + 1] - cycles[k]
        denom = delta / d_solv + expo[k] / k_sei
        j_sei_k = (-F_CONST * consts.c_solv_mol_m3 / denom).abs()
        int_sei = (j_sei_k * w[k]).sum() * area_eff / 3600.0    # А·ч
        int_pl = jpl_int[k] * area_eff / 3600.0                  # А·ч
        dq_dead = consts.beta_dead * int_pl
        dq_lli = int_sei + dq_dead
        if resid is not None:
            feats_k = torch.stack([cell.i_a[k].abs().sum() / 3600.0,
                                   cell.phi_n[k].mean(), cell.phi_n[k].min(),
                                   (cell.phi_n[k] < 0).double().mean(), delta * 1e9])
            dq_lli = dq_lli + resid(feats_k, cell.feats)
        delta = delta + int_sei * 3600.0 * consts.v_sei_m3_mol / (F_CONST * area_eff)
        q_new = (q_li_pred[-1] - gap * dq_lli).clamp(min=1e-6)
        q_li_pred.append(q_new)
        deltas.append(delta)
        r_pred.append(r0 + delta * rho_sei / 1.54e-4)
    return {
        "q_li": torch.stack(q_li_pred),
        "r_total": torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in r_pred]),
        "delta_sei": torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in deltas]),
        "cycles": cycles,
    }


def mech_increments(cell: CellData, z: torch.Tensor,
                    consts: DegradationConsts,
                    temp_k: float = T_REF) -> tuple[np.ndarray, np.ndarray]:
    """Пошаговые механические приращения при заданном z.

    Возвращает ``(dq_lli[k], ddelta[k])`` для шагов k→k+1 уже с учётом
    разрыва циклов, массивы длины B-1 (А·ч и м). Используется как входной
    канал переходной модели: «что дала бы чистая механика на этом шаге».
    """
    area_eff = 10.0 ** float(z[4])
    w = trapz_weights(cell.t_s)
    d_solv = consts.d_solv_m2_s * (10.0 ** float(z[1]))
    k_sei = consts.k_sei_m_s * (10.0 ** float(z[0]))
    j0_pl = consts.j0_pl_a_m2 * (10.0 ** float(z[2]))
    a_sei = consts.alpha_sei * F_CONST / (8.314462618 * temp_k)
    a_pl = consts.alpha_pl * F_CONST / (8.314462618 * temp_k)
    expo = torch.exp(a_sei * (cell.phi_n - consts.u_sei_v)).clamp(max=1e6)
    expp = torch.exp((a_pl * cell.phi_n).clamp(max=50.0))
    expm = torch.exp((-a_pl * cell.phi_n).clamp(max=50.0))
    j_pl = torch.where(cell.phi_n < 0.0, j0_pl * (expp - expm),
                       torch.zeros_like(cell.phi_n))
    jpl_int = (j_pl.abs() * w).sum(dim=1)
    delta, dq_out, dd_out = consts.delta_sei0_m, [], []
    for k in range(len(cell.cycles) - 1):
        gap = cell.cycles[k + 1] - cell.cycles[k]
        denom = delta / d_solv + expo[k] / k_sei
        j_sei_k = (-F_CONST * consts.c_solv_mol_m3 / denom).abs()
        int_sei = float((j_sei_k * w[k]).sum()) * area_eff / 3600.0
        int_pl = float(jpl_int[k]) * area_eff / 3600.0
        dd = int_sei * 3600.0 * consts.v_sei_m3_mol / (F_CONST * area_eff)
        dq_out.append(gap * (int_sei + consts.beta_dead * int_pl))
        dd_out.append(gap * dd)
        delta += dd
    return np.asarray(dq_out), np.asarray(dd_out)


def loss_rollout(cell: CellData, pred: dict, lam_r: float = 0.5,
                 lam_mono: float = 10.0) -> torch.Tensor:
    """Потеря прогона: относительные ошибки q_li и r + монотонность q_li.

    Поцикловое R шумно (идентификация даёт выбросы вплоть до ~мОм) —
    потеря по R считается в логарифмах и только по правдоподобным точкам
    (0.2·медиана < r < 5·медиана).
    """
    q_true = torch.tensor(cell.q_li, dtype=torch.float64)
    r_true = torch.tensor(cell.r_total, dtype=torch.float64)
    l_q = (((pred["q_li"] - q_true) / q_true[0]) ** 2).mean()
    med = r_true.median().clamp_min(1e-3)
    ok = (r_true > 0.2 * med) & (r_true < 5.0 * med)
    if ok.sum() >= 3:
        l_r = ((pred["r_total"][ok].clamp_min(1e-6).log()
                - r_true[ok].clamp_min(1e-6).log()) ** 2).mean()
    else:
        l_r = torch.zeros((), dtype=torch.float64)
    dq = pred["q_li"][1:] - pred["q_li"][:-1]
    l_mono = (torch.relu(dq) ** 2).mean() * 1e4
    return l_q + lam_r * l_r + lam_mono * l_mono


def loss_increments(cell: CellData, pred: dict) -> torch.Tensor:
    """Потеря по приращениям: Δq_li между идентифицированными циклами.

    Уровни q_li шумны (выбросы идентификации), приращения устойчивее.
    """
    dq_p = pred["q_li"][1:] - pred["q_li"][:-1]
    dq_t = torch.tensor(np.diff(cell.q_li), dtype=torch.float64)
    d = dq_p - dq_t
    scale = cell.q_li[0] ** 2
    hub = torch.where(d.abs() <= 1e-4, 0.5 * d ** 2,
                      1e-4 * (d.abs() - 0.5e-4))
    return hub.mean() / scale


def train_encoder(cells: list[CellData], consts: DegradationConsts,
                  epochs: int = 300, lr: float = 3e-3,
                  lam_z: float = 0.02, lam_inc: float = 1.0,
                  verbose: bool = True) -> CellEncoder:
    """Обучение кодирующей сети на наборе элементов (full-batch Adam).

    Потеря — сумма по уровням q_li (связь состояния) и приращениям
    (кинетика), плюс слабая регуляризация z.
    """
    enc = CellEncoder().double()
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        tot = torch.zeros(())
        for cell in cells:
            z = enc(cell.feats)
            pred = rollout_state(cell, z, consts)
            tot = tot + loss_rollout(cell, pred) \
                + lam_inc * loss_increments(cell, pred) \
                + lam_z * (z ** 2).mean()
        tot = tot / len(cells)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        opt.step()
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"ep {ep:4d}: loss {float(tot.detach()):.5f}")
    return enc
