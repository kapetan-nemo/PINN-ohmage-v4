"""Импорт датасета LIR2025H (Mendeley m8w8sjk3vm) в каноническую схему BDF.

Источник: ``data/raw/lir2025/Battery Data.zip`` — файлы
``Original Bat Data/batN/M.csv`` (один CSV на цикл, выгрузка Landt CT3001A:
Cycle ID, Step Name, Time(h:min:s.ms), Voltage(V), Current(mA), ...).
Ток разряда отрицателен — совпадает с соглашением Aurora
(«заряд положителен»), знак не инвертируется.

Результат:
  ``data/raw/lir2025/lir2025h__batNN/lir2025h__batNN.bdf.parquet``
      столбцы t_s (с), i_a (А), v_v (В), cycle, temp_c (°C, из политики)
  ``data/params/lir2025_protocol.json``
      по элементу: v_max/v_min (В), i_chg_a, c_rate, temp_c —
      Charging Policy.xlsx + наблюдаемая нижняя отсечка разряда.

Запуск: ``python scripts/import_lir2025.py``
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "lir2025"
OUT = ROOT / "data" / "raw" / "lir2025"
PARAMS = ROOT / "data" / "params"


def _hms_to_s(text: str) -> float:
    """``h:mm:ss.ms`` → секунды."""
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _read_cycle_csv(buf: bytes) -> pl.DataFrame:
    df = pl.read_csv(
        io.BytesIO(buf),
        columns=["Cycle ID", "Step Name", "Time(h:min:s.ms)",
                 "Voltage(V)", "Current(mA)", "Realtime"],
    )
    return df


def charging_policy(path: Path) -> dict[int, dict]:
    """Charging Policy.xlsx → {№ элемента: параметры заряда}."""
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.worksheets[0]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        out[int(row[0])] = {
            "i_chg_ma": float(row[1]),
            "c_rate": float(row[2]),
            "v_max": float(row[3]),
            "temp_c": float(row[4]),
        }
    return out


def main() -> None:
    zip_path = SRC / "Battery Data.zip"
    policy = charging_policy(SRC / "Charging Policy.xlsx")
    proto: dict[str, dict] = {}

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if re.search(r"bat(\d+)/(\d+)\.csv$", m)]
        # batN/M.csv → (bat, cycle)
        def key(m: str) -> tuple[int, int]:
            g = re.search(r"bat(\d+)/(\d+)\.csv$", m)
            return int(g.group(1)), int(g.group(2))
        members.sort(key=key)
        bats: dict[int, list[str]] = {}
        for m in members:
            bats.setdefault(key(m)[0], []).append(m)
        print(f"элементов: {len(bats)}, файлов: {len(members)}")

        for bat, files in sorted(bats.items()):
            cid = f"lir2025h__bat{bat:02d}"
            frames = []
            t_off = 0.0      # глобальное смещение t_s по элементу
            prev_end = 0.0
            for m in files:
                df = _read_cycle_csv(zf.read(m))
                if df.height == 0:
                    continue
                # ВНИМАНИЕ: колонка Time(h:min:s.ms) обнуляется на каждом
                # шаге протокола (Rest/CCCV/CC_DChg), а не на цикл — для
                # монотонного времени внутри цикла берём номер строки:
                # выгрузка Landt идёт с кадансом ровно 1 Гц.
                t = np.arange(df.height, dtype=float) + t_off
                t_off = float(t[-1]) + 1.0
                if t_off <= prev_end:  # защита от пустых циклов
                    t_off = prev_end + 1.0
                prev_end = t_off
                frames.append(pl.DataFrame({
                    "cycle": np.full(df.height, key(m)[1], dtype=np.int64),
                    "t_s": t,
                    "i_a": df["Current(mA)"].to_numpy() * 1e-3,
                    "v_v": df["Voltage(V)"].to_numpy(),
                    "temp_c": np.full(df.height,
                                      policy.get(bat, {}).get("temp_c", 25.0)),
                }))
            if not frames:
                continue
            cell = pl.concat(frames)
            out_dir = OUT / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            cell.write_parquet(out_dir / f"{cid}.bdf.parquet")

            dch = cell.filter(pl.col("i_a") < -1e-4)["v_v"].min()
            pol = policy.get(bat, {})
            proto[cid] = {
                "v_max_main": pol.get("v_max", 4.2),
                # нижняя отсечка — по факту данных (~2.75 В)
                "v_min_main": round(float(dch) + 0.02, 3)
                              if dch is not None else 2.75,
                "i_chg_ma": pol.get("i_chg_ma"),
                "c_rate": pol.get("c_rate"),
                "temp_c": pol.get("temp_c"),
            }
            print(f"  {cid}: {cell.height} точек, "
                  f"циклов {cell['cycle'].max()}, "
                  f"v_min≈{proto[cid]['v_min_main']} В")

    PARAMS.mkdir(parents=True, exist_ok=True)
    (PARAMS / "lir2025_protocol.json").write_text(
        json.dumps(proto, indent=1, ensure_ascii=False))
    print("протоколы →", PARAMS / "lir2025_protocol.json")


if __name__ == "__main__":
    sys.exit(main())
