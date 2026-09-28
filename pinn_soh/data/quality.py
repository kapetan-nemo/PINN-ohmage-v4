"""Статистика циклов и контроль качества элементов набора Aurora.

Две группы результатов:

* поцикловые величины — интегрированные ёмкости заряда и разряда
  (столбцов накопленной ёмкости в данных нет), кулоновская эффективность,
  границы напряжения, доля пауз, признаки артефактов;
* сводные величины элемента — число формовочных циклов, опорная ёмкость
  ``Q0`` первого стабильного цикла, траектория SOH, циклы достижения уровней
  SOH, итоговый статус контроля качества.

Соглашение о знаке тока приводится вызывающей стороной: после умножения
``i_a`` на возвращаемый :func:`pinn_soh.data.bdf_loader.detect_current_sign`
положительный ток соответствует заряду.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

# Порог «отсутствия тока» для выделения пауз, А.
REST_CURRENT_A = 1e-4
# Максимальный интервал интегрирования, с; большие разрывы в ёмкость не входят.
MAX_INTEGRATION_DT_S = 120.0
# Допустимые границы напряжения элемента, В.
V_MIN_VALID, V_MAX_VALID = 0.0, 5.0
# Скачок напряжения без изменения тока — признак артефакта, В.
V_JUMP_ARTEFACT = 1.0
# Разрыв времени — признак артефакта, с.
T_GAP_ARTEFACT_S = 24 * 3600.0
# Критерии стабильной фазы кулоновской эффективности.
CE_LOW, CE_HIGH = 0.90, 1.02
# Пороги уровней SOH для постобработки.
DEFAULT_SOH_LEVELS = (0.95, 0.90, 0.85, 0.80)


def per_cycle_stats(df: pl.DataFrame, current_sign: int = 1) -> pl.DataFrame:
    """Вычисляет поцикловые величины по кадру внутренней схемы.

    Возвращает таблицу со столбцами: ``cycle, t0, t1, n, duration_s,
    q_chg_ah, q_dchg_ah, ce, i_max_a, v_min, v_max, v_mean, temp_mean,
    rest_frac, n_t_gaps, max_dt_s, n_v_oor, n_v_jumps, first_v, last_v``.
    Ёмкости — интегралы |I| dt по полуциклам заряда и разряда, А·ч;
    интервалы длиннее ``MAX_INTEGRATION_DT_S`` в интеграл не входят.
    """
    work = (
        df.sort("t_s")
        .with_columns(
            (pl.col("i_a") * current_sign).alias("i_chg"),
            pl.col("t_s").diff().over("cycle").alias("dt"),
        )
        .with_columns(
            pl.col("dt").clip(0.0, MAX_INTEGRATION_DT_S).fill_null(0.0).alias("dt_int"),
        )
    )
    return (
        work.group_by("cycle")
        .agg(
            pl.col("t_s").min().alias("t0"),
            pl.col("t_s").max().alias("t1"),
            pl.len().alias("n"),
            (pl.col("i_chg").clip(lower_bound=0.0) * pl.col("dt_int")).sum().alias("q_chg_ah"),
            ((-pl.col("i_chg")).clip(lower_bound=0.0) * pl.col("dt_int")).sum().alias("q_dchg_ah"),
            pl.col("i_chg").abs().max().alias("i_max_a"),
            pl.col("v_v").min().alias("v_min"),
            pl.col("v_v").max().alias("v_max"),
            pl.col("v_v").mean().alias("v_mean"),
            pl.col("temp_c").mean().alias("temp_mean"),
            ((pl.col("i_a").abs() < REST_CURRENT_A).cast(pl.Float64) * pl.col("dt").fill_null(0.0))
            .sum()
            .alias("rest_s"),
            (pl.col("dt") > T_GAP_ARTEFACT_S).sum().alias("n_t_gaps"),
            pl.col("dt").max().alias("max_dt_s"),
            ((pl.col("v_v") < V_MIN_VALID) | (pl.col("v_v") > V_MAX_VALID)).sum().alias("n_v_oor"),
            (
                (pl.col("v_v").diff().abs() > V_JUMP_ARTEFACT)
                & (pl.col("i_a").diff().abs() < REST_CURRENT_A)
            )
            .sum()
            .alias("n_v_jumps"),
            pl.col("v_v").first().alias("first_v"),
            pl.col("v_v").last().alias("last_v"),
        )
        .with_columns(
            (pl.col("t1") - pl.col("t0")).alias("duration_s"),
            (pl.col("q_chg_ah") / 3600.0).alias("q_chg_ah"),
            (pl.col("q_dchg_ah") / 3600.0).alias("q_dchg_ah"),
        )
        .with_columns(
            (pl.col("q_dchg_ah") / pl.col("q_chg_ah")).alias("ce"),
            (pl.col("rest_s") / (pl.col("t1") - pl.col("t0"))).alias("rest_frac"),
        )
        .sort("cycle")
    )


def count_formation_cycles(protocol: list) -> int | None:
    """Число формовочных циклов по протоколу: ``repeat`` первой группы ``workflow``.

    Формовочная группа в протоколе Aurora — первая повторяющаяся группа после
    предварительного заряда и паузы; основная группа имеет большой ``repeat``.
    Возвращает ``None``, если протокол пуст или группа не определена.
    """
    for step in protocol:
        if getattr(step, "mode", None) == "workflow" and step.repeat:
            return int(step.repeat)
    return None


def formation_cycles_from_data(cycles: pl.DataFrame, min_cycles: int = 1, max_cycles: int = 20) -> int:
    """Число формовочных циклов по данным: начальные циклы с током меньше
    половины медианного тока основной части. Используется как резервный путь,
    когда протокол в метаданных недоступен."""
    if cycles.height < min_cycles + 2:
        return 0
    main = cycles.slice(min_cycles)
    med_i = main["i_max_a"].median()
    if med_i is None or med_i <= 0:
        return 0
    n = 0
    for row in cycles.slice(0, max_cycles).iter_rows(named=True):
        if row["i_max_a"] < 0.5 * med_i:
            n += 1
        else:
            break
    return n


@dataclass
class CellReport:
    """Сводный отчёт качества по одному элементу."""

    cell_id: str
    current_sign: int = 1
    n_cycles: int = 0
    formation_cycles: int = 0
    duration_days: float | None = None
    q0_ah: float | None = None
    nominal_capacity_ah: float | None = None
    soh_first: float | None = None
    soh_last: float | None = None
    soh_min: float | None = None
    levels_reached: dict[str, int | None] = field(default_factory=dict)
    ce_median_stable: float | None = None
    temp_range: tuple[float | None, float | None] = (None, None)
    n_artefact_cycles: int = 0
    artefact_cycles: list[int] = field(default_factory=list)
    has_time_gap: bool = False
    non_monotonic_time: bool = False
    status: str = "ok"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    protocol_summary: dict = field(default_factory=dict)


def _protocol_summary(meta) -> dict:
    """Краткое описание протокола для отчёта и разбиения."""
    summary: dict = {}
    if meta is None:
        return summary
    workflows = [s for s in meta.protocol if s.mode == "workflow"]
    if len(workflows) >= 2:
        main = workflows[-1]
        summary["main_repeat"] = main.repeat
        limits = [
            s.voltage_limit_v for s in main.steps if s.voltage_limit_v is not None
        ]
        currents = [s.current_a for s in main.steps if s.current_a is not None]
        if limits:
            summary["v_max_main"] = max(limits)
            summary["v_min_main"] = min(limits)
        if currents:
            summary["i_main_a"] = max(currents, key=abs)
    return summary


def build_cell_report(
    cell_id: str,
    df: pl.DataFrame,
    meta=None,
    levels: tuple[float, ...] = DEFAULT_SOH_LEVELS,
) -> tuple[CellReport, pl.DataFrame]:
    """Формирует отчёт качества и поцикловую таблицу одного элемента.

    ``df`` — кадр внутренней схемы :func:`load_bdf`; ``meta`` —
    :class:`CellMetadata` или ``None``. Возвращает пару
    (отчёт, таблица циклов с добавленным столбцом ``soh``).
    """
    from pinn_soh.data.bdf_loader import detect_current_sign

    rep = CellReport(cell_id=cell_id)
    rep.current_sign = detect_current_sign(df)
    rep.nominal_capacity_ah = getattr(meta, "nominal_capacity_ah", None)
    rep.protocol_summary = _protocol_summary(meta)

    t_diff = df["t_s"].diff().drop_nulls()
    rep.non_monotonic_time = bool((t_diff < 0).any())
    rep.has_time_gap = bool((t_diff > T_GAP_ARTEFACT_S).any())

    cycles = per_cycle_stats(df, rep.current_sign)
    rep.n_cycles = int(cycles.filter(pl.col("cycle") >= 1).height)

    f_meta = count_formation_cycles(getattr(meta, "protocol", []) or [])
    rep.formation_cycles = f_meta if f_meta is not None else formation_cycles_from_data(cycles)

    main = cycles.filter(pl.col("cycle") >= rep.formation_cycles + 1)
    rep.duration_days = (
        round(float(df["t_s"].max() - df["t_s"].min()) / 86400.0, 2)
        if df.height
        else None
    )
    temps = df["temp_c"].drop_nulls()
    rep.temp_range = (
        (float(temps.min()), float(temps.max())) if temps.len() else (None, None)
    )

    if main.height == 0:
        rep.status = "artefact"
        rep.reasons.append("no_main_cycles")
        return rep, cycles

    rep.q0_ah = float(main["q_dchg_ah"][0]) if main["q_dchg_ah"][0] else None
    cycles = cycles.with_columns(
        (pl.col("q_dchg_ah") / rep.q0_ah).alias("soh") if rep.q0_ah else pl.lit(None).alias("soh")
    )
    main = cycles.filter(pl.col("cycle") >= rep.formation_cycles + 1)

    rep.soh_first = float(main["soh"][0]) if rep.q0_ah else None
    rep.soh_last = float(main["soh"][-1]) if rep.q0_ah else None
    rep.soh_min = float(main["soh"].min()) if rep.q0_ah else None
    for level in levels:
        hit = main.filter(pl.col("soh") < level)
        rep.levels_reached[f"{level:.2f}"] = int(hit["cycle"][0]) if hit.height else None

    stable = main.filter(pl.col("cycle") >= rep.formation_cycles + 3)
    rep.ce_median_stable = float(stable["ce"].median()) if stable.height else None

    # Артефактные циклы (поцикловое исключение): недопустимое напряжение,
    # скачки без изменения тока, отклонение SOH более 3 п.п. от медианы
    # соседних циклов (одиночные провалы и выбросы ёмкости).
    bad = cycles.filter((pl.col("n_v_oor") > 0) | (pl.col("n_v_jumps") > 0))
    artefact_set = {int(c) for c in bad["cycle"].to_list()}
    if rep.q0_ah and main.height > 6:
        soh = main["soh"].to_list()
        cyc_ids = main["cycle"].to_list()
        for i in range(2, len(soh) - 2):
            neigh = sorted(
                x for j, x in enumerate(soh) if j != i and abs(j - i) <= 2 and x is not None
            )
            if soh[i] is not None and neigh and abs(soh[i] - neigh[len(neigh) // 2]) > 0.03:
                artefact_set.add(cyc_ids[i])
    rep.artefact_cycles = sorted(artefact_set)
    rep.n_artefact_cycles = len(rep.artefact_cycles)

    # Устойчивый ступенчатый рост SOH более 8 п.п. (медиана трёх последующих
    # циклов выше медианы трёх предыдущих) после первых десяти основных
    # циклов — не основание для исключения: по разведочному анализу такие
    # ступени в наборе Aurora соответствуют смене тока циклирования либо
    # эпизодам восстановления ёмкости и являются физически реальными.
    # Фиксируется как флаг ``sustained_soh_step_up@<цикл>``; естественные
    # циклические колебания ёмкости в наборе достигают ±4 п.п.
    if rep.q0_ah and main.height > 16:
        soh = main["soh"].to_list()
        cyc_ids = main["cycle"].to_list()
        durs = main["duration_s"].to_list()
        for i in range(13, len(soh) - 3):
            before = soh[i - 3 : i]
            after = soh[i : i + 3]
            if None in before + after:
                continue
            if sorted(after)[1] - sorted(before)[1] > 0.08:
                rep.flags.append(f"sustained_soh_step_up@{cyc_ids[i]}")
                db = sorted(durs[i - 3 : i])[1]
                da = sorted(durs[i : i + 3])[1]
                if db and abs(da - db) / db > 0.2:
                    rep.flags.append(f"protocol_rate_change@{cyc_ids[i]}")
                break

    # Итоговый статус: правила группы A (артефакты → исключение).
    if rep.non_monotonic_time:
        rep.reasons.append("non_monotonic_time")
    if rep.has_time_gap and not rep.non_monotonic_time:
        rep.reasons.append("time_gap_gt_24h")
    if rep.n_cycles < 100:
        rep.reasons.append("lt_100_cycles")
    if rep.q0_ah is not None and rep.nominal_capacity_ah and rep.q0_ah < 0.5 * rep.nominal_capacity_ah:
        rep.reasons.append("q0_lt_half_nominal")
    if rep.ce_median_stable is not None and not (CE_LOW <= rep.ce_median_stable <= CE_HIGH):
        rep.reasons.append(f"ce_out_of_range:{rep.ce_median_stable:.3f}")

    artefact_frac = rep.n_artefact_cycles / max(rep.n_cycles, 1)
    if rep.n_artefact_cycles > 0 and artefact_frac > 0.5:
        rep.reasons.append(f"artefact_cycles_frac:{artefact_frac:.2f}")

    if rep.reasons:
        rep.status = "artefact"

    return rep, cycles
