"""Разбор метаданных элемента (JSON-LD, онтология BattINFO).

Файлы ``*.metadata.json`` набора Empa Aurora имеют структуру ``@graph`` с узлом
``BatteryTest``, который содержит ``hasTestObject`` (``CoinCell``: электроды,
электролит, сепаратор, корпус) и ``hasMeasurementParameter`` (протокол
циклирования в виде цепочки задач по ссылкам ``hasNext``). Разбор построен на
этой схеме, но все ветви выполнены устойчиво: при отсутствии ожидаемых ключей
функция не падает, а откладывает нераспознанные узлы в ``raw_unmatched``.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Соответствие типа задачи протокола режиму шага.
_TASK_MODE = {
    "charging": "cc_charge",
    "hold": "cv_charge",
    "discharging": "cc_discharge",
    "resting": "rest",
    "rest": "rest",
    "iterativeworkflow": "workflow",
}

# Множители приведения единиц времени к секундам.
_TIME_UNIT_S = {"hour": 3600.0, "minute": 60.0, "second": 1.0}


@dataclass
class ProtocolStep:
    """Один шаг протокола циклирования.

    ``inputs`` сохраняет исходные величины шага вида ``{тип: (значение,
    единица)}``, а ``steps`` — вложенную последовательность для шагов типа
    ``workflow`` (повторяющаяся группа).
    """

    mode: str = "unknown"
    current_a: float | None = None
    c_rate: float | None = None
    voltage_limit_v: float | None = None
    current_cutoff_a: float | None = None
    duration_s: float | None = None
    repeat: int | None = None
    inputs: dict[str, tuple[float | None, str | None]] = field(default_factory=dict)
    steps: list["ProtocolStep"] = field(default_factory=list)


@dataclass
class CellMetadata:
    """Извлечённые свойства элемента из файла метаданных."""

    cell_id: str = ""
    cathode_material: str | None = None
    anode_material: str | None = None
    nominal_capacity_ah: float | None = None
    electrode_masses: dict[str, float] = field(default_factory=dict)
    electrode_areas: dict[str, float] = field(default_factory=dict)
    electrolyte: str | None = None
    protocol: list[ProtocolStep] = field(default_factory=list)
    raw_unmatched: dict[str, Any] = field(default_factory=dict)


def _types(node: Any) -> list[str]:
    """Нормализованный список ``@type`` узла."""
    value = node.get("@type") if isinstance(node, dict) else None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _has_type(node: Any, *names: str) -> bool:
    """Проверяет совпадение ``@type`` узла с образцами без учёта регистра."""
    lowered = {t.lower() for t in _types(node)}
    return any(n.lower() in lowered for n in names)


def _number(node: Any) -> float | None:
    """Числовое значение измеренной величины (``hasNumericalPart.hasNumberValue``)."""
    if not isinstance(node, dict):
        return float(node) if isinstance(node, (int, float)) else None
    part = node.get("hasNumericalPart")
    if isinstance(part, dict):
        value = part.get("hasNumberValue")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _unit(node: Any) -> str | None:
    """Единица измерения величины (``hasMeasurementUnit``)."""
    value = node.get("hasMeasurementUnit") if isinstance(node, dict) else None
    return value if isinstance(value, str) else None


def _as_list(value: Any) -> list[Any]:
    """Приводит значение к списку."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _measured(node: Any) -> list[dict]:
    """Список измеренных свойств узла (``hasMeasuredProperty``)."""
    props = node.get("hasMeasuredProperty") if isinstance(node, dict) else None
    return [p for p in _as_list(props) if isinstance(p, dict)]


def _material_name(active_material: Any) -> str | None:
    """Имя активного материала: формула, метка либо ``@type``."""
    if not isinstance(active_material, dict):
        return None
    formula = active_material.get("molecularFormula")
    if isinstance(formula, dict):
        text = formula.get("rdfs:comment") or formula.get("rdfs:label")
        if isinstance(text, str):
            return text.strip()
    elif isinstance(formula, str):
        return formula.strip()
    for key in ("rdfs:comment", "rdfs:label", "schema:name"):
        text = active_material.get(key)
        if isinstance(text, str):
            return text.strip()
        if isinstance(text, list) and text and isinstance(text[0], str):
            return text[0].strip()
    types = _types(active_material)
    return types[0].strip() if types else None


def _area_from_diameter_cm2(diameter_mm: float | None) -> float | None:
    """Площадь дискового электрода по диаметру в миллиметрах, см²."""
    if not diameter_mm:
        return None
    return math.pi * (diameter_mm / 20.0) ** 2


def _to_amperes(value: float | None, unit: str | None, area_cm2: float | None) -> float | None:
    """Приводит величину тока к амперам; плотность тока — через площадь электрода."""
    if value is None or unit is None:
        return value
    unit_l = unit.lower()
    if "per" in unit_l or "persquare" in unit_l.replace("squarecentimetre", ""):
        # Плотность тока (мА/см²): умножаем на площадь электрода.
        if area_cm2 is None:
            return None
        if "milli" in unit_l:
            return value * area_cm2 / 1000.0
        if "micro" in unit_l:
            return value * area_cm2 / 1e6
        return value * area_cm2
    if "milli" in unit_l:
        return value / 1000.0
    return value


def _to_seconds(value: float | None, unit: str | None) -> float | None:
    """Приводит длительность к секундам."""
    if value is None:
        return None
    unit_l = (unit or "").lower()
    for name, factor in _TIME_UNIT_S.items():
        if name in unit_l:
            return value * factor
    return value


def _find_nodes(node: Any, predicate, results: list | None = None) -> list:
    """Рекурсивно собирает узлы документа, удовлетворяющие предикату."""
    if results is None:
        results = []
    if predicate(node):
        results.append(node)
    if isinstance(node, dict):
        for value in node.values():
            _find_nodes(value, predicate, results)
    elif isinstance(node, list):
        for value in node:
            _find_nodes(value, predicate, results)
    return results


def _parse_task(task: Any, area_cm2: float | None, depth: int = 0) -> list[ProtocolStep]:
    """Разбирает цепочку задач протокола по ссылкам ``hasNext``."""
    steps: list[ProtocolStep] = []
    seen: set[int] = set()
    node = task
    while isinstance(node, dict) and id(node) not in seen and depth < 64:
        seen.add(id(node))
        step = ProtocolStep()
        for t in _types(node):
            mode = _TASK_MODE.get(t.lower())
            if mode:
                step.mode = mode
                break
        for inp in _as_list(node.get("hasInput")):
            if not isinstance(inp, dict):
                continue
            input_types = {t.lower() for t in _types(inp)}
            value, unit = _number(inp), _unit(inp)
            # В паре видов типа «UpperVoltageLimit, TerminationQuantity»
            # меткой служит специфичный тип.
            specific = sorted(input_types - {"terminationquantity"})
            label = specific[0] if specific else (sorted(input_types)[0] if input_types else "unknown")
            step.inputs[label] = (value, unit)
            if "electriccurrentdensity" in input_types or "electriccurrent" in input_types:
                step.current_a = _to_amperes(value, unit, area_cm2)
            elif "uppervoltagelimit" in input_types or "lowervoltagelimit" in input_types or "voltage" in input_types:
                step.voltage_limit_v = value
            elif "lowercurrentdensitylimit" in input_types or "currentlimit" in input_types:
                step.current_cutoff_a = _to_amperes(value, unit, area_cm2)
            elif "duration" in input_types:
                step.duration_s = _to_seconds(value, unit)
            elif "numberofiterations" in input_types:
                step.repeat = int(value) if value is not None else None
            elif "c-rate" in input_types or "crate" in input_types:
                step.c_rate = value
        if step.mode == "workflow":
            step.steps = _parse_task(node.get("hasTask"), area_cm2, depth + 1)
        steps.append(step)
        node = node.get("hasNext")
    return steps


def _electrode_summary(electrode: Any, tag: str, meta: CellMetadata) -> tuple[float | None, float | None]:
    """Извлекает диаметр, удельную и поверхностную ёмкость электрода."""
    diameter_mm = areal_mah_cm2 = None
    for prop in _measured(electrode):
        value, unit = _number(prop), _unit(prop)
        unit_l = (unit or "").lower()
        if _has_type(prop, "Diameter") and "millim" in unit_l:
            diameter_mm = value
        elif _has_type(prop, "RatedCapacity") and "squarecentimetre" in unit_l.replace("-", ""):
            if value is not None:
                areal_mah_cm2 = value / 1000.0 if "milli" in unit_l else value
    if diameter_mm:
        meta.electrode_areas[f"{tag}_cm2"] = _area_from_diameter_cm2(diameter_mm)
    coating = electrode.get("hasCoating") if isinstance(electrode, dict) else None
    for prop in _measured(coating):
        value, unit = _number(prop), _unit(prop)
        if value is None:
            continue
        type_name = _types(prop)[0] if _types(prop) else "property"
        meta.electrode_masses[f"{tag}.{type_name}:{unit}"] = value
    active = coating.get("hasActiveMaterial") if isinstance(coating, dict) else None
    for prop in _measured(active):
        value, unit = _number(prop), _unit(prop)
        if value is None:
            continue
        type_name = _types(prop)[0] if _types(prop) else "property"
        meta.electrode_masses[f"{tag}.active.{type_name}:{unit}"] = value
    return diameter_mm, areal_mah_cm2


def _electrolyte_name(electrolyte: Any) -> str | None:
    """Строковое описание состава электролита по компонентам и их долям."""
    parts: list[str] = []

    def constituents(node: Any) -> list[dict]:
        return [c for c in _as_list(node.get("hasConstituent")) if isinstance(c, dict)] if isinstance(node, dict) else []

    solvent = electrolyte.get("hasSolvent") if isinstance(electrolyte, dict) else None
    solute = electrolyte.get("hasSolute") if isinstance(electrolyte, dict) else None
    groups = constituents(solvent)
    groups += constituents(solute)
    for additive in _as_list((solute or {}).get("hasAdditive")):
        groups += constituents(additive)
    for component in groups:
        name = _types(component)[0] if _types(component) else "component"
        props = _measured(component)
        frac = _number(props[0]) if props else None
        unit = _unit(props[0]) if props else ""
        suffix = "M" if "mol" in (unit or "").lower() else "%"
        parts.append(f"{name} ({frac}{suffix})" if frac is not None else name)
    return ", ".join(parts) if parts else None


def parse_metadata(path: str | Path) -> CellMetadata:
    """Разбирает файл ``*.metadata.json`` в структуру :class:`CellMetadata`."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    meta = CellMetadata(cell_id=path.name.split(".metadata.json")[0])

    # Элемент: первый узел типа CoinCell (вне веток @reverse полуэлементных тестов).
    cells = _find_nodes(document, lambda n: _has_type(n, "CoinCell"))
    cell = cells[0] if cells else None
    pos = cell.get("hasPositiveElectrode") if isinstance(cell, dict) else None
    neg = cell.get("hasNegativeElectrode") if isinstance(cell, dict) else None
    pos_area = neg_area = None
    if isinstance(pos, dict):
        meta.cathode_material = _material_name((pos.get("hasCoating") or {}).get("hasActiveMaterial"))
        _, pos_areal = _electrode_summary(pos, "positive", meta)
        pos_area = meta.electrode_areas.get("positive_cm2")
        if pos_areal and pos_area:
            meta.nominal_capacity_ah = pos_areal * pos_area
    if isinstance(neg, dict):
        meta.anode_material = _material_name((neg.get("hasCoating") or {}).get("hasActiveMaterial"))
        _electrode_summary(neg, "negative", meta)
        neg_area = meta.electrode_areas.get("negative_cm2")
    if isinstance(cell, dict) and isinstance(cell.get("hasElectrolyte"), dict):
        meta.electrolyte = _electrolyte_name(cell["hasElectrolyte"])

    # Протокол: hasMeasurementParameter верхнего уровня BatteryTest.
    area = pos_area or neg_area
    tests = _find_nodes(
        document,
        lambda n: _has_type(n, "BatteryTest") and isinstance(n, dict) and "hasMeasurementParameter" in n,
    )
    for test in tests:
        param = test.get("hasMeasurementParameter")
        if isinstance(param, dict) and "hasTask" in param:
            meta.protocol = _parse_task(param.get("hasTask"), area)
            break

    # Нераспознанные разделы верхнего уровня элемента.
    if isinstance(cell, dict):
        for key, value in cell.items():
            if key not in (
                "@type", "hasPositiveElectrode", "hasNegativeElectrode",
                "hasElectrolyte", "schema:productID",
            ):
                meta.raw_unmatched[key] = value
    return meta
