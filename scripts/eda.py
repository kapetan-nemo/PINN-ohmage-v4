"""Разведочный анализ набора Empa Aurora (этап 2 плана).

Обходит все элементы, для каждого строит поцикловую таблицу и отчёт качества
(:func:`pinn_soh.data.quality.build_cell_report`). Результаты:

* ``configs/cell_quality.json`` — статус и причины по каждому элементу;
* ``configs/split.json`` — разбиение элементов на обучающую, проверочную и
  контрольную выборки со стратификацией по составу и группе качества
  (метки состава используются только для разбиения);
* ``data/processed/cycles/<cell>.parquet`` — поцикловые таблицы с SOH;
* ``reports/eda_summary.json`` — сводная статистика набора;
* на печать — компактная сводка для лабораторного журнала.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_soh.data.bdf_loader import list_cells, load_bdf  # noqa: E402
from pinn_soh.data.metadata import parse_metadata  # noqa: E402
from pinn_soh.data.quality import build_cell_report  # noqa: E402

RAW = ROOT / "data" / "raw" / "aurora"
CYCLES_DIR = ROOT / "data" / "processed" / "cycles"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"


def chemistry_group(meta) -> str:
    """Группа состава для стратификации (не вход модели)."""
    name = (meta.cathode_material or "").lower() if meta else ""
    if "lifepo4" in name or "lfp" in name:
        return "lfp"
    if "ni0.83" in name or "ni83" in name:
        return "nmc811"
    if "ni" in name:
        return "nmc622"
    return "unknown"


def main() -> None:
    cells = list_cells(RAW)
    print(f"элементов: {len(cells)}")
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    quality: dict[str, dict] = {}
    summaries: list[dict] = []
    t_start = time.time()
    for i, cf in enumerate(cells):
        try:
            meta = parse_metadata(cf.metadata_path) if cf.metadata_path else None
            df = load_bdf(cf.parquet_path)
            rep, cycles = build_cell_report(cf.cell_id, df, meta)
        except Exception as exc:  # не прерывать проход из-за одного файла
            quality[cf.cell_id] = {"status": "artefact", "reasons": [f"load_error:{exc}"]}
            print(f"  [{i + 1}/{len(cells)}] {cf.cell_id}: ОШИБКА {exc}")
            continue
        cycles.write_parquet(CYCLES_DIR / f"{cf.cell_id}.parquet")
        group = chemistry_group(meta)
        quality[cf.cell_id] = {
            "status": rep.status,
            "reasons": rep.reasons,
            "flags": rep.flags,
            "n_cycles": rep.n_cycles,
            "formation_cycles": rep.formation_cycles,
            "chemistry_group": group,
            "v_max_main": rep.protocol_summary.get("v_max_main"),
            "soh_last": rep.soh_last,
        }
        summaries.append(
            {
                "cell_id": cf.cell_id,
                "group": group,
                "status": rep.status,
                "current_sign": rep.current_sign,
                "n_cycles": rep.n_cycles,
                "formation": rep.formation_cycles,
                "days": rep.duration_days,
                "q0_mah": round(rep.q0_ah * 1000, 4) if rep.q0_ah else None,
                "q_nom_mah": round(rep.nominal_capacity_ah * 1000, 4)
                if rep.nominal_capacity_ah
                else None,
                "soh_first": rep.soh_first,
                "soh_last": rep.soh_last,
                "soh_min": rep.soh_min,
                **{f"n_{k}": v for k, v in rep.levels_reached.items()},
                "ce_med": rep.ce_median_stable,
                "t_lo": rep.temp_range[0],
                "t_hi": rep.temp_range[1],
                "n_artefact": rep.n_artefact_cycles,
                "gap24": rep.has_time_gap,
                "nonmono": rep.non_monotonic_time,
                "v_max_main": rep.protocol_summary.get("v_max_main"),
                "i_main_ma": round(rep.protocol_summary["i_main_a"] * 1000, 4)
                if rep.protocol_summary.get("i_main_a")
                else None,
            }
        )
        if (i + 1) % 20 == 0 or i + 1 == len(cells):
            print(f"  [{i + 1}/{len(cells)}] {time.time() - t_start:.0f} с")

    # --- сводная статистика ---
    s = pl.DataFrame(summaries, infer_schema_length=None)

    # Группа B: преждевременный физический отказ — цикл достижения SOH 0.80
    # ниже нижнего дециля своей группы по v_max основного цикла.
    thresholds: dict[float | None, float] = {}
    for vmax in s["v_max_main"].unique().to_list():
        sub = s.filter(
            (pl.col("v_max_main") == vmax)
            & (pl.col("status") != "artefact")
            & pl.col("n_0.80").is_not_null()
        )
        if sub.height >= 10:
            thresholds[vmax] = float(sub["n_0.80"].quantile(0.1))
    n_early = 0
    for row in s.iter_rows(named=True):
        if row["status"] == "artefact" or row["n_0.80"] is None:
            continue
        thr = thresholds.get(row["v_max_main"])
        if thr is not None and row["n_0.80"] < thr:
            quality[row["cell_id"]]["status"] = "early_failure"
            quality[row["cell_id"]]["reasons"] = [f"soh_0.80_at_cycle:{row['n_0.80']}"]
            n_early += 1
    s = s.with_columns(
        pl.col("cell_id")
        .map_elements(lambda c: quality[c]["status"], return_dtype=pl.String)
        .alias("status")
    )

    (CONFIGS / "cell_quality.json").write_text(json.dumps(quality, indent=1))

    s.write_parquet(REPORTS / "eda_cells.parquet")
    ok = s.filter(pl.col("status") != "artefact")
    levels = ["0.95", "0.90", "0.85", "0.80"]
    summary = {
        "cells_total": s.height,
        "status_counts": dict(Counter(s["status"].to_list())),
        "chemistry_counts": dict(Counter(s["group"].to_list())),
        "sign_counts": dict(Counter(str(x) for x in s["current_sign"].to_list())),
        "formation_counts": dict(Counter(str(x) for x in s["formation"].to_list())),
        "v_max_counts": dict(Counter(str(x) for x in s["v_max_main"].to_list())),
        "cycles_median": s["n_cycles"].median(),
        "days_median": s["days"].median(),
        "temp_min": s["t_lo"].min(),
        "temp_max": s["t_hi"].max(),
        "soh_last_quantiles_ok": {
            q: ok["soh_last"].quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        if ok.height
        else {},
        "levels_reached_count": {
            lv: int(ok[f"n_{lv}"].drop_nulls().len()) for lv in levels
        },
        "ce_median": s["ce_med"].median(),
    }
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # --- разбиение по элементам: 80/10/10, стратификация по составу и статусу ---
    import random

    rng = random.Random(20260925)
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for row in s.iter_rows(named=True):
        buckets[(row["group"], "artefact" if row["status"] == "artefact" else "ok")].append(
            row["cell_id"]
        )
    split = {"train": [], "val": [], "test": []}
    for ids in buckets.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(0.1 * n))
        n_val = max(1, round(0.1 * n))
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test : n_test + n_val]
        split["train"] += ids[n_test + n_val :]
    for key in split:
        split[key] = sorted(split[key])
    split["artefact_excluded"] = sorted(
        row["cell_id"] for row in s.iter_rows(named=True) if row["status"] == "artefact"
    )
    (CONFIGS / "split.json").write_text(json.dumps(split, indent=1))
    print(
        f"split: train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} excluded={len(split['artefact_excluded'])}"
    )


if __name__ == "__main__":
    main()
