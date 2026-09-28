"""Проверки чтения файлов BDF и определения знака тока."""

import numpy as np
import polars as pl

from pinn_soh.data.bdf_loader import INTERNAL_COLUMNS, detect_current_sign, list_cells, load_bdf


def _make_bdf_frame(n: int = 200, sign: int = 1) -> pl.DataFrame:
    t = np.arange(n, dtype=float)
    # Заряд при положительном токе (sign=+1): напряжение растёт.
    current = np.where(t % 20 < 10, 0.01 * sign, -0.01 * sign)
    voltage = 3.0 + np.cumsum(np.where(t % 20 < 10, 1.0, -1.0) * 0.001)
    return pl.DataFrame(
        {
            "test_time_second": t,
            "current_ampere": current,
            "voltage_volt": voltage,
            "cycle_count": np.ones(n, dtype=int),
            "step_count": np.where(t % 20 < 10, 1, 2),
            "temperature_t1_celsius": np.full(n, 25.0),
            "charging_capacity_ah": np.abs(current).cumsum() / 3600,
            "discharging_capacity_ah": np.zeros(n),
        }
    )


def test_load_bdf_renames_columns(tmp_path):
    path = tmp_path / "cell.bdf.parquet"
    _make_bdf_frame().write_parquet(path)
    frame = load_bdf(path)
    for column in INTERNAL_COLUMNS:
        assert column in frame.columns
    assert frame["t_s"].to_list() == list(range(200))
    assert "test_time_second" not in frame.columns


def test_load_bdf_alternate_column_names(tmp_path):
    # Фактическая разметка parquet набора Aurora.
    frame = _make_bdf_frame().rename(
        {
            "test_time_second": "test_time_millisecond",
            "cycle_count": "cycle_dimensionless",
            "temperature_t1_celsius": "ambient_temperature_celsius",
        }
    ).with_columns(pl.col("test_time_millisecond") * 1000)
    path = tmp_path / "cell2.bdf.parquet"
    frame.write_parquet(path)
    loaded = load_bdf(path)
    assert loaded["t_s"].to_list() == list(range(200))
    assert detect_current_sign(frame) == 1


def test_load_bdf_missing_columns_filled_with_null(tmp_path):
    path = tmp_path / "partial.bdf.parquet"
    _make_bdf_frame().drop(["temperature_t1_celsius", "charging_capacity_ah"]).write_parquet(path)
    frame = load_bdf(path)
    assert frame["temp_c"].null_count() == frame.height
    assert frame["q_chg_ah"].null_count() == frame.height


def test_detect_current_sign_positive_and_negative(tmp_path):
    assert detect_current_sign(_make_bdf_frame(sign=1)) == 1
    assert detect_current_sign(_make_bdf_frame(sign=-1)) == -1


def test_list_cells_pairs_parquet_and_metadata(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    _make_bdf_frame(10).write_parquet(nested / "empa__ccid000001.bdf.parquet")
    (nested / "empa__ccid000001.metadata.json").write_text("{}")
    # Файл без парных метаданных.
    _make_bdf_frame(10).write_parquet(tmp_path / "empa__ccid000002.bdf.parquet")
    cells = list_cells(tmp_path)
    assert [c.cell_id for c in cells] == ["empa__ccid000001", "empa__ccid000002"]
    assert cells[0].metadata_path is not None
    assert cells[1].metadata_path is None
