"""Прогнозный тест: первые K циклов → SOH до конца жизни.

Для каждого элемента и каждой длины истории K выполняется:

1. идентификация состояния только по циклам ``<= K`` (префикс);
2. калибровка кинетики ``z`` на извлечённой траектории префикса;
3. рекурсивный прогноз до последнего фактического цикла;
4. сравнение с измеренной разрядной ёмкостью и построение графиков.

    .venv/bin/python scripts/forecast_soh.py \
        --cells empa__ccid000208 --history 5 10 15 25 50 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.data.bdf_loader import list_cells, load_bdf  # noqa: E402
from pinn_soh.data.metadata import parse_metadata  # noqa: E402
from pinn_soh.data.preprocess import build_pseudo_ocv, preprocess_cell  # noqa: E402
from pinn_soh.data.quality import build_cell_report  # noqa: E402
from pinn_soh.baselines import fit_empirical, predict_empirical  # noqa: E402
from pinn_soh.eval.metrics import threshold_cycle  # noqa: E402
from pinn_soh.models.encoder import CellEncoder, build_features  # noqa: E402
from pinn_soh.physics.degradation import DegradationConsts  # noqa: E402
from pinn_soh.predict.forecast import (  # noqa: E402
    fit_z_prefix, forecast, forecast_ensemble,
)
from pinn_soh.train.stage1_extract_state import (  # noqa: E402
    identify_cell, load_anchors, load_pretrained_ocv, refine_per_cycle,
)

PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "forecast"


def measured_soh(cycles: pl.DataFrame, first_main: int,
                 artefact: set[int] | None = None):
    """Измеренная SOH: разрядная ёмкость / ёмкость первого основного цикла.

    Отбрасываются: циклы с длительностью < 70% медианы (оборванные записи,
    у 38% элементов последний цикл усечён) и артефактные циклы из отчёта
    качества (выбросы напряжения/SOH) — иначе точка пришивки и метрики
    загрязняются артефактами.
    """
    cc = cycles.sort("cycle")
    ids = cc["cycle"].to_numpy()
    q = cc["q_dchg_ah"].to_numpy()
    dur = cc["duration_s"].to_numpy() \
        if "duration_s" in cc.columns else np.full(len(ids), 1.0)
    m = ids >= first_main
    ids, q, dur = ids[m], q[m], dur[m]
    med_dur = np.median(dur[5:]) if len(dur) > 10 else np.median(dur)
    ok = dur >= 0.7 * med_dur
    if artefact:
        ok = ok & ~np.isin(ids, list(artefact))
    if ok.any():
        ids, q = ids[ok], q[ok]
    q0 = q[0]
    return ids, q / q0, q0


def norm_soh_handoff(r, f, cy_t, soh_t):
    """SOH прогноза пришивается к измеренной на границе истории."""
    if len(f.cycles) == 0:
        raise RuntimeError("пустой горизонт прогноза")
    k_h = float(r.cycles[-1])
    s_h = float(np.interp(k_h, cy_t, soh_t))
    q0 = f.q_dch_ah[0]
    if not np.isfinite(q0) or q0 <= 0:
        # член-судьба умер на первом шаге прогноза — вся кривая нулевая
        f.soh = np.zeros_like(f.q_dch_ah)
    else:
        f.soh = f.q_dch_ah / q0 * s_h
    return f


def load_r_fate_stats():
    """Популяционная статистика взрыва R по полножизненным идентификациям.

    Возвращает список записей ``(cell_id, r_at_100, r_max, k_cross)``:
    R на ~100-м цикле (группировка приора), максимум R за жизнь и цикл
    пересечения R=100 Ом. Записи поэлементные — прогнозируемый элемент
    исключается из приора на месте (см. ``fate_scenarios``): иначе
    собственный R_max ячейки участвует в квантилях её же сценариев.
    """
    recs = []
    d = ROOT / "checkpoints" / "stage1"
    if not d.exists():
        return recs
    for f in sorted(d.glob("*.json")):
        try:
            j = json.loads(f.read_text())
            cy = np.asarray(j["cycles"], float)
            r = np.asarray(j["r_total_ohm"], float)
        except Exception:
            continue
        if len(cy) < 8 or cy[0] > 50:
            continue
        i100 = int(np.argmin(np.abs(cy - 100)))
        if cy[i100] > 130:
            continue
        xc = cy[r > 100.0]
        recs.append((str(j.get("cell_id") or f.stem), float(r[i100]),
                     float(r.max()),
                     float(xc[0]) if len(xc) else None))
    return recs


def fate_scenarios(r_prefix, n, stats, exclude_id=None):
    """N судеб ``(q_li_scale, (r_max, k_mid, w))`` по условным квантилям.

    Члены покрывают квантильную сетку R_max своей группы («R@100 ≤ 30 Ом»
    / «> 30 Ом» по префиксному R прогнозируемого элемента); точка взрыва
    и лог-рассеяние q_li на стыке (σ≈0.1 декс — разброс prefix/full
    идентификаций) распределяются по сетке со сдвигом фазы.
    ``exclude_id`` — прогнозируемый элемент: его полножизненная
    идентификация не должна входить в его же приор.
    """
    grp, kc = [], []
    for cid, r100, rmax, kx in stats:
        if cid == exclude_id:
            continue
        if (r100 > 30.0) == (r_prefix > 30.0):
            grp.append(rmax)
        if kx is not None:
            kc.append(kx)
    if len(grp) < 5 or len(kc) < 5:
        return []
    from scipy.special import ndtri
    out = []
    for j in range(n):
        u = (j + 0.5) / n
        r_max = float(np.quantile(grp, u))
        k_mid = float(np.quantile(kc, (u + 0.37) % 1.0))
        qs = float(10.0 ** (0.10 * ndtri((u + 0.71) % 1.0)))
        out.append((qs, (r_max, k_mid, 60.0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--history", type=int, nargs="+",
                    default=[5, 10, 15, 25, 50, 100])
    ap.add_argument("--iters-b", type=int, default=500)
    ap.add_argument("--refine", type=int, default=200)
    ap.add_argument("--fstride", type=int, default=5)
    ap.add_argument("--kinetic-cap", action="store_true",
                    help="ёмкость прогноза — по симуляции разряда до v_min")
    ap.add_argument("--r-boost", type=float, default=1.0,
                    help="масштаб персистентности на R-канале")
    ap.add_argument("--r-film", action="store_true",
                    help="R из толщины SEI-плёнки вместо выученного канала")
    ap.add_argument("--r-qli-gamma", type=float, default=0.0,
                    help="рост R обратно пропорционально инвентарю: "
                         "R = R_гр·(q_li_гр/q_li)^γ; 0 — выкл")
    ap.add_argument("--ocv-clamp", action="store_true",
                    help="в прогнозе OCV не экстраполируется за θ∈[0,1]")
    ap.add_argument("--encoder", default="population",
                    help="'population' — медиана z_population.json; "
                         "путь к чекпоинту encoder; '' — без априора")
    ap.add_argument("--ensemble", type=int, default=0,
                    help="число членов ансамбля для полосы неопределённости")
    ap.add_argument("--ensemble-ks", type=int, nargs="+", default=None,
                    help="для каких K рисовать полосу (по умолчанию все)")
    ap.add_argument("--ident-ensemble", type=int, default=0,
                    help="число членов ансамбля идентификаций "
                         "(разные iters_b/stride → разные решения "
                         "обратной задачи)")
    ap.add_argument("--r-fates", type=int, default=0,
                    help="число членов ансамбля судеб: квантильная сетка "
                         "сценариев взрыва R и рассеяния q_li на стыке "
                         "(популяционный приор, переидентификации нет)")
    ap.add_argument("--transition",
                    default=str(ROOT / "checkpoints" / "stage3" / "transition.pt"),
                    help="переходная модель состояния; '' — выкл")
    ap.add_argument("--tag", default=None,
                    help="метка прогона → подкаталог reports/forecast/<tag>/; "
                         "по умолчанию — метка времени")
    args = ap.parse_args()

    tag = args.tag or time.strftime("run_%Y%m%d_%H%M")
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_meta.json"
    try:
        prev = json.loads(meta_path.read_text()) \
            if meta_path.exists() else {}
    except (json.JSONDecodeError, OSError):   # гонка параллельных шардов
        prev = {}
    cells_meta = sorted(set(prev.get("cells", [])) | set(args.cells))
    meta_path.write_text(json.dumps({
        "tag": tag, "cells": cells_meta,
        "history": sorted(set(prev.get("history", [])) | set(args.history)),
        "z_prior": args.encoder or "none", "transition": args.transition,
        "iters_b": args.iters_b, "refine": args.refine,
        "fstride": args.fstride, "ensemble": args.ensemble,
        "r_fates": args.r_fates,
        "started": prev.get("started", time.strftime("%Y-%m-%d %H:%M:%S"))},
        indent=1))
    print("каталог прогона:", out_dir, flush=True)
    # Aurora + импортированные наборы (LIR2025H и др.) — один индекс
    cells_idx: dict = {}
    for ds_root in (ROOT / "data" / "raw").iterdir():
        if ds_root.is_dir():
            cells_idx.update(
                {c.cell_id: c for c in list_cells(ds_root)})
    # сайдкар-протоколы для наборов без JSON-LD метаданных (LIR2025H)
    proto_sidecar = {}
    for sc in (ROOT / "data" / "params").glob("*_protocol.json"):
        proto_sidecar.update(json.loads(sc.read_text()))
    ocv_n, ocv_p = load_pretrained_ocv(ROOT / "checkpoints" / "stage0")
    anchors = load_anchors(ROOT / "data" / "params" / "latent_anchors.json")
    kin = ROOT / "data" / "params" / "kinetics_OKane2022.json"
    consts = DegradationConsts.from_json(kin) if kin.exists() else DegradationConsts()
    enc = None
    z_pop = None
    if args.encoder == "population":
        zp = ROOT / "data" / "params" / "z_population.json"
        if zp.exists():
            z_pop = torch.tensor(json.loads(zp.read_text())["median"],
                                 dtype=torch.float64)
            print("априор — популяционная медиана z:",
                  np.round(z_pop.numpy(), 2), flush=True)
    elif args.encoder and Path(args.encoder).exists():
        enc = CellEncoder().double()
        enc.load_state_dict(torch.load(args.encoder, weights_only=False)["encoder"])
        enc.eval()
        print("encoder загружен:", args.encoder, flush=True)
    trans = None
    if args.transition and Path(args.transition).exists():
        from pinn_soh.models.transition import TransitionNet
        trans = TransitionNet().double()
        # strict=False: старые чекпоинты без параметра β
        trans.load_state_dict(
            torch.load(args.transition, weights_only=False)["transition"],
            strict=False)
        trans.eval()
        print("переходная модель загружена:", args.transition, flush=True)
    quality = json.loads((ROOT / "configs" / "cell_quality.json").read_text())
    r_fate_stats = load_r_fate_stats() if args.r_fates else None

    summary = []
    for cid in args.cells:
        cf = cells_idx[cid]
        meta = (parse_metadata(cf.metadata_path)
                if cf.metadata_path and cf.metadata_path.exists() else None)
        df = load_bdf(cf.parquet_path)
        rep, cycles = build_cell_report(cid, df, meta)
        if cid in proto_sidecar:          # LIR2025H: отсечки из сайдкара
            rep.protocol_summary.update(proto_sidecar[cid])
        proc_path = PROC / f"{cid}.parquet"
        if proc_path.exists():
            pdf = pl.read_parquet(proc_path)
        else:
            proc = preprocess_cell(cid, df, cycles, rep.formation_cycles,
                                   set(rep.artefact_cycles), rep.current_sign)
            proc.df.write_parquet(proc_path)
            pdf = proc.df
        area_m2 = ((meta.electrode_areas.get("positive_cm2") or 1.539)
                   if meta is not None else 1.539) * 1e-4
        v_max = rep.protocol_summary.get("v_max_main") or 4.2
        v_min = rep.protocol_summary.get("v_min_main") or 2.5
        pseudo = build_pseudo_ocv(df, rep.formation_cycles)
        last_cycle = int(pdf["cycle"].max())
        first_main = rep.formation_cycles + 1
        cy_t, soh_t, q_ref = measured_soh(
            cycles, first_main, set(rep.artefact_cycles))
        print(f"\n=== {cid}: циклов до {last_cycle}, отсечка {v_max} В "
              f"(SOH_кон {soh_t[-1]:.3f}) ===", flush=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for idx, K in enumerate(args.history):
            color = cmap(idx % 10)
            t0 = time.time()

            def run_one(iters_b=args.iters_b, stride_id=20):
                r = identify_cell(
                    cid, pdf, cycles, ocv_n, ocv_p, anchors,
                    formation_cycles=rep.formation_cycles,
                    area_m2=area_m2, stride=stride_id,
                    iters_a=200, iters_b=iters_b,
                    v_min=v_min, v_max=v_max, pseudo=pseudo,
                    max_cycle=first_main + K - 1, verbose=False)
                if args.refine:
                    r = refine_per_cycle(r, pdf, cycles, ocv_n, ocv_p,
                                         area_m2=area_m2, iters=args.refine)
                md = dict(r.__dict__)
                md["v_max_main"] = quality.get(cid, {}).get("v_max_main")
                ac_ = None
                for nm, a in anchors.get("positive", {}).items():
                    if nm == r.anchor_name:
                        ac_ = torch.tensor(a["code"])
                feats_ = build_features(md, ac_)
                z0_ = z_pop if enc is None else enc(feats_)
                z_ = fit_z_prefix(r, pdf, consts, iters=150, z0=z0_,
                                  area_m2=area_m2)
                if trans is None:
                    feats_ = None
                f = forecast(r, pdf, ocv_n, ocv_p, consts, z_,
                             cycle_end=last_cycle, v_min=v_min, v_max=v_max,
                             stride=args.fstride, area_m2=area_m2,
                             kinetic_cap=args.kinetic_cap,
                             r_boost=args.r_boost, r_film=args.r_film,
                             r_qli_gamma=args.r_qli_gamma,
                             ocv_clamp=args.ocv_clamp,
                             trans=trans, feats=feats_)
                return r, norm_soh_handoff(r, f, cy_t, soh_t), z_, feats_

            try:
                res, fc, z, feats_main = run_one()
                k_hand = float(res.cycles[-1])
                s_hand = float(np.interp(k_hand, cy_t, soh_t))
                st = np.interp(fc.cycles, cy_t, soh_t)
                rmse = float(np.sqrt(np.mean((fc.soh - st) ** 2)))
                k80_p = threshold_cycle(fc.cycles, fc.soh, 0.8)
                k80_t = threshold_cycle(cy_t, soh_t, 0.8)
                e80 = (k80_p - k80_t) if (k80_p and k80_t) else None
                summary.append({
                    "cell": cid, "K": K, "n_ident": len(res.cycles),
                    "v_rmse_mv": res.v_rmse_mv, "soh_rmse": rmse,
                    "k80_pred": k80_p, "k80_true": k80_t, "k80_err": e80,
                    "z": [float(x) for x in z],
                    "gains": (fc.gains.tolist() if fc.gains is not None else None),
                    "floors": (fc.floors.tolist() if fc.floors is not None else None),
                    "runtime_s": time.time() - t0,
                })
                ax.plot(fc.cycles, fc.soh, "-", lw=1.8, color=color,
                        label=f"K={K} (RMSE {rmse:.3f})")
                ax.plot(k_hand, s_hand, "o", ms=10, color=color,
                        mec="k", mew=1.2, zorder=5)
                # эмпирический базис на том же префиксе — сравнение
                mpre = cy_t <= k_hand
                if mpre.sum() >= 3:
                    coef = fit_empirical(cy_t[mpre], soh_t[mpre])
                    kgrid = np.arange(k_hand, last_cycle + 1, args.fstride)
                    s_emp = predict_empirical(coef, kgrid, cy_t[mpre][0])
                    ax.plot(kgrid, np.clip(s_emp, 0, None), "--", lw=1.0,
                            color=color, alpha=0.55)
                    st_e = np.interp(kgrid, cy_t, soh_t)
                    rmse_a = float(np.sqrt(np.mean((s_emp - st_e) ** 2)))
                    summary[-1]["soh_rmse_empirical"] = rmse_a
                if args.ensemble and (args.ensemble_ks is None
                                      or K in args.ensemble_ks):
                    # та же конфигурация, что у основного прогноза —
                    # иначе полоса неопределённости отвечает другой модели
                    ens = forecast_ensemble(
                        res, pdf, ocv_n, ocv_p, consts, z,
                        cycle_end=last_cycle, v_min=v_min, v_max=v_max,
                        stride=args.fstride, area_m2=area_m2,
                        n_members=args.ensemble,
                        kinetic_cap=args.kinetic_cap, r_boost=args.r_boost,
                        r_film=args.r_film, r_qli_gamma=args.r_qli_gamma,
                        ocv_clamp=args.ocv_clamp, trans=trans,
                        feats=feats_main)
                    ax.fill_between(ens["cycles"],
                                    ens["soh_p10"] * s_hand,
                                    ens["soh_p90"] * s_hand,
                                    color=color, alpha=0.10, lw=0)
                ens_members = None
                need_band = (args.ensemble_ks is None
                             or K in args.ensemble_ks)
                mem_c, mem_s = [fc.cycles], [fc.soh]
                if args.ident_ensemble > 0 and need_band:
                    # ось разнообразия — гиперпараметры обратной задачи:
                    # identify_cell детерминирован при фиксированных
                    # (iters_b, stride), разные пары → разные локальные
                    # решения; каждый член пришит к измеренной SOH границы
                    # порядок перемешан по осям: малые N уже покрывают
                    # и iters_b, и stride; первый член — основной прогон
                    grid = [(400, 15), (600, 25), (500, 15), (400, 25),
                            (600, 20), (400, 20), (500, 25), (600, 15)]
                    for ib, st_ in grid[1:args.ident_ensemble]:
                        try:
                            _, fm, _, _ = run_one(iters_b=ib, stride_id=st_)
                            mem_c.append(fm.cycles)
                            mem_s.append(fm.soh)
                        except Exception as e2:
                            print(f"    член ({ib},{st_}): сбой {e2}",
                                  flush=True)
                if args.r_fates > 0 and need_band:
                    # ось судеб: взрыв R и уровень q_li на стыке невидимы/
                    # слабо ограничены на префиксе — члены покрывают
                    # популяционные квантили на ТОЙ ЖЕ идентификации
                    for qs, rf in fate_scenarios(res.r_total_ohm[-1],
                                                 args.r_fates,
                                                 r_fate_stats,
                                                 exclude_id=cid):
                        try:
                            fm = forecast(
                                res, pdf, ocv_n, ocv_p, consts, z,
                                cycle_end=last_cycle, v_min=v_min,
                                v_max=v_max, stride=args.fstride,
                                area_m2=area_m2,
                                kinetic_cap=args.kinetic_cap,
                                r_boost=args.r_boost, r_film=args.r_film,
                                r_qli_gamma=args.r_qli_gamma,
                                ocv_clamp=args.ocv_clamp,
                                trans=trans, feats=feats_main,
                                q_li_start=res.q_li_ah[-1] * qs,
                                r_fate=rf)
                            norm_soh_handoff(res, fm, cy_t, soh_t)
                            mem_c.append(fm.cycles)
                            mem_s.append(fm.soh)
                        except Exception as e2:
                            print(f"    судьба {rf}: сбой {e2}", flush=True)
                if need_band and trans is not None and fc.gains is not None:
                    # ось LAM: ранняя осадка λ на префиксе неоднозначна —
                    # у здоровых она транзиентная (λ возвращается к ~1),
                    # у умирающих персистентна. Два полярных члена на той же
                    # идентификации: без SNR-стягивания (LAM идёт) и с
                    # задавленными λ-усилителями (LAM насыщается)
                    g_lam_off = np.asarray(fc.gains, float).copy()
                    g_lam_off[1:3] = 0.3
                    lam_variants = (dict(lam_snr=False),
                                    dict(gains=g_lam_off))
                    for kw_ in lam_variants:
                        try:
                            fm = forecast(
                                res, pdf, ocv_n, ocv_p, consts, z,
                                cycle_end=last_cycle, v_min=v_min,
                                v_max=v_max, stride=args.fstride,
                                area_m2=area_m2,
                                kinetic_cap=args.kinetic_cap,
                                r_boost=args.r_boost, r_film=args.r_film,
                                r_qli_gamma=args.r_qli_gamma,
                                ocv_clamp=args.ocv_clamp,
                                trans=trans, feats=feats_main, **kw_)
                            norm_soh_handoff(res, fm, cy_t, soh_t)
                            mem_c.append(fm.cycles)
                            mem_s.append(fm.soh)
                        except Exception as e2:
                            print(f"    LAM-ветка {kw_}: сбой {e2}",
                                  flush=True)
                if len(mem_c) > 1:
                    kg = np.arange(
                        min(c[0] for c in mem_c),
                        max(c[-1] for c in mem_c) + 1, args.fstride)
                    sm = np.stack([np.interp(kg, c, s,
                                             left=np.nan, right=np.nan)
                                   for c, s in zip(mem_c, mem_s)])
                    med = np.nanmedian(sm, axis=0)
                    ens_members = (kg, med,
                                   np.nanpercentile(sm, 10, axis=0),
                                   np.nanpercentile(sm, 90, axis=0), sm)
                    st_m = np.interp(kg, cy_t, soh_t)
                    ok_m = np.isfinite(med)
                    rmse_med = float(np.sqrt(np.mean(
                        (med[ok_m] - st_m[ok_m]) ** 2)))
                    summary[-1]["soh_rmse_median"] = rmse_med
                    for s_ in sm[1:]:
                        ax.plot(kg, s_, "-", color=color, lw=0.7,
                                alpha=0.3)
                    ax.fill_between(kg, ens_members[2], ens_members[3],
                                    color=color, alpha=0.12, lw=0)
                    ax.plot(kg, med, ":", lw=2.2, color=color,
                            label=f"K={K} медиана (RMSE {rmse_med:.3f})")
                extra = {}
                if ens_members is not None:
                    kg_, med_, p10_, p90_, sm_ = ens_members
                    extra = dict(cycles_ens=kg_, soh_ens_median=med_,
                                 soh_ens_p10=p10_, soh_ens_p90=p90_,
                                 soh_ens_members=sm_)
                np.savez_compressed(
                    out_dir / f"{cid}_K{K}.npz",
                    cycles_pred=fc.cycles, soh_pred=fc.soh,
                    cycles_true=cy_t, soh_true=soh_t,
                    q_li=fc.q_li_ah, r_total=fc.r_total_ohm,
                    q_dch=fc.q_dch_ah,
                    theta_n0=fc.theta_n0, theta_p0=fc.theta_p0,
                    lam_n=fc.lam_n, lam_p=fc.lam_p, q_dch_raw=fc.q_dch_raw,
                    q_win=fc.q_win, q_sim=fc.q_sim,
                    delta_sei=fc.delta_sei_m, **extra)
                print(f"  K={K:3d}: идент {len(res.cycles)} циклов, "
                      f"V RMSE {res.v_rmse_mv:.0f} мВ, SOH RMSE {rmse:.3f}, "
                      f"k80 {e80 if e80 is not None else float('nan'):+.0f}, "
                      f"{time.time()-t0:.0f} с", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  K={K}: СБОЙ {e}", flush=True)
                summary.append({"cell": cid, "K": K, "error": str(e)})
        ax.plot(cy_t, soh_t, "k-", lw=3.0, label="факт", zorder=4)
        for lv in (0.95, 0.90, 0.85, 0.80):
            ax.axhline(lv, color="r", ls=":", lw=0.6, alpha=0.4)
        ax.grid(True, which="both", ls="-", lw=0.4, alpha=0.5)
        ax.set_xlabel("цикл")
        ax.set_ylabel("SOH")
        ax.set_title(f"{cid}: прогноз SOH по первым K циклам "
                     f"(точка — граница истории)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cid}_soh_forecast.png", dpi=120)
        plt.close(fig)

    # сводка отдельно по каждому элементу — для параллельных запусков
    for cid in {s.get("cell") for s in summary if "cell" in s}:
        rows = [s for s in summary if s.get("cell") == cid]
        (out_dir / f"summary_{cid}.json").write_text(
            json.dumps({
                "config": {
                    "z_prior": args.encoder or "none",
                    "iters_b": args.iters_b, "refine": args.refine,
                    "fstride": args.fstride,
                    "ensemble": args.ensemble, "tag": tag,
                },
                "rows": rows}, indent=1, ensure_ascii=False))
    print("\nсохранено:", out_dir)


if __name__ == "__main__":
    main()
