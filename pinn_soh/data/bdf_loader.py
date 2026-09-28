"""Чтение файлов BDF набора данных Empa Aurora в каноническую схему.

В наборе данных встречаются два набора имён столбцов: спецификация BDF
(``test_time_second``, ``cycle_count``, …) и фактическая разметка parquet
(``test_time_millisecond``, ``cycle_dimensionless``, …). Отображение задаётся
списком кандидатов с коэффициентом приведения единиц.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

# Внутренний столбец → [(исходное имя, множитель к внутренней единице)].
# t_s — секунды; i_a — амперы; v_v — вольты; temp_c — °C; ёмкости — А·ч.
BDF_COLUMN_CANDIDATES: dict[str, list[tuple[str, float]]] = {
    "t_s": [("test_time_second", 1.0), ("test_time_millisecond", 1e-3)],
    "i_a": [("current_ampere", 1.0), ("i_a", 1.0)],
    "v_v": [("voltage_volt", 1.0), ("v_v", 1.0)],
    "cycle": [("cycle_count", 1.0), ("cycle_dimensionless", 1.0)],
    "step": [("step_count", 1.0)],
    "temp_c": [("temperature_t1_celsius", 1.0), ("ambient_temperature_celsius", 1.0)],
    "q_chg_ah": [("charging_capacity_ah", 1.0)],
    "q_dchg_ah": [("discharging_capacity_ah", 1.0)],
}

# Обратная совместимость: первый кандидат каждого столбца.
BDF_COLUMN_MAP: dict[str, str] = {
    candidates[0][0]: column for column, candidates in BDF_COLUMN_CANDIDATES.items()
}

INTERNAL_COLUMNS = list(BDF_COLUMN_CANDIDATES)


@dataclass
class CellFiles:
    """Пути к файлам одного элемента в распакованном RO-Crate."""

    cell_id: str
    parquet_path: Path
    metadata_path: Path | None


def load_bdf(path: str | Path) -> pl.DataFrame:
    """Читает один файл ``*.bdf.parquet`` и приводит столбцы к внутренней схеме.

    При совпадении нескольких кандидатов используется первый по порядку;
    величины приводятся к внутренним единицам заданным множителем.
    Отсутствующие столбцы дополняются значениями null, нераспознанные
    сохраняются без переименования.
    """
    frame = pl.read_parquet(path)
    return normalize_bdf_frame(frame)


def normalize_bdf_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Приводит уже загруженный кадр к внутренней схеме (аналог :func:`load_bdf`)."""
    for column, candidates in BDF_COLUMN_CANDIDATES.items():
        for source, scale in candidates:
            if source in frame.columns:
                if source != column:
                    frame = frame.rename({source: column})
                if scale != 1.0:
                    frame = frame.with_columns((pl.col(column) * scale).alias(column))
                break
        else:
            if column not in frame.columns:
                frame = frame.with_columns(pl.lit(None).alias(column))
    return frame


def list_cells(root: str | Path) -> list[CellFiles]:
    """Рекурсивно обходит каталог и собирает пары файлов по каждому элементу."""
    root = Path(root)
    cells: list[CellFiles] = []
    for parquet_path in sorted(root.rglob("*.bdf.parquet")):
        cell_id = parquet_path.name.split(".bdf.parquet")[0]
        metadata_name = cell_id + ".metadata.json"
        metadata_path = parquet_path.with_name(metadata_name)
        if not metadata_path.exists():
            matches = list(root.rglob(metadata_name))
            metadata_path = matches[0] if matches else None  # type: ignore[assignment]
        cells.append(CellFiles(cell_id=cell_id, parquet_path=parquet_path, metadata_path=metadata_path))
    return cells


def detect_current_sign(df: pl.DataFrame, min_current_a: float = 1e-4) -> int:
    """Определяет знак соглашения о токе по соотношению знаков I и dV/dt.

    Возвращает +1, если положительный ток соответствует заряду (росту
    напряжения), и -1 в противном случае. Принимает кадр как во внутренней
    схеме, так и с исходными именами столбцов BDF.
    """
    df = normalize_bdf_frame(df)
    sign = (
        df.select("t_s", "i_a", "v_v")
        .filter(pl.col("i_a").abs() > min_current_a)
        .sort("t_s")
        .select(
            (
                pl.corr(
                    pl.col("i_a").sign(),
                    pl.col("v_v").diff().sign(),
                )
            ).alias("corr")
        )
        .item()
    )
    if sign is None:
        return 1
    return 1 if sign > 0 else -1
