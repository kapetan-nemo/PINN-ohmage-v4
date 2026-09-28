"""Постобработка прогнозных траекторий: пересечение порогов SOH.

После рекурсивного моделирования произвольного числа циклов траектория
SOH передаётся сюда для извлечения циклов пересечения запрошенных
порогов (например 0.95/0.90/0.85/0.80) и остаточного ресурса.
"""
from __future__ import annotations

import numpy as np

from .metrics import threshold_cycle

DEFAULT_LEVELS = (0.95, 0.90, 0.85, 0.80)


def crossing_report(cycles: np.ndarray, soh: np.ndarray,
                    levels=DEFAULT_LEVELS,
                    soh_reference: float | None = None) -> dict:
    """Отчёт по порогам: цикл пересечения и остаточный ресурс (RUL).

    ``soh_reference`` — цикл, от которого считается RUL; по умолчанию —
    первый цикл траектории. Возвращает словарь вида::

        {
          "0.8": {"cross_cycle": 842.3, "rul_cycles": 512.3, "reached": True},
          "0.85": {"cross_cycle": None, "rul_cycles": None, "reached": False},
          ...
        }
    """
    cycles = np.asarray(cycles, float)
    k0 = float(soh_reference) if soh_reference is not None else float(cycles[0])
    out = {}
    for lv in levels:
        kc = threshold_cycle(cycles, soh, lv)
        out[f"{lv:g}"] = {
            "cross_cycle": kc,
            "rul_cycles": (kc - k0) if kc is not None else None,
            "reached": kc is not None,
        }
    return out


def multi_cell_table(records: list[dict], levels=DEFAULT_LEVELS) -> "list[dict]":
    """Сводная таблица пересечений по набору элементов.

    ``records`` — список словарей ``{"cell_id", "cycles", "soh"}``;
    возвращает строки ``{"cell_id", "level", "cross_cycle", ...}``.
    """
    rows = []
    for rec in records:
        rep = crossing_report(rec["cycles"], rec["soh"], levels=levels)
        for lv, r in rep.items():
            rows.append({"cell_id": rec["cell_id"], "level": float(lv), **r})
    return rows
