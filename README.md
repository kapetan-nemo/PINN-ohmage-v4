# pinn_soh

Физически информированная нейросетевая модель деградации дисковых литий-ионных
элементов: прогноз напряжения и состояния элемента (SOH) по профилю тока
на произвольный горизонт циклов. Обучение — на наборе данных Empa Aurora.

## Установка

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
```

Набор параметров с полуэлементными кривыми (PyBaMM) — при необходимости:

```bash
uv pip install -e ".[params]"
```

## Загрузка набора данных

```bash
pinn-soh-download            # либо: python -m pinn_soh.data.download
```

Архив `Dataset-rocrate.zip` (2,5 ГБ) загружается в `data/raw/` с докачкой по
Range при обрыве соединения, проверкой контрольной суммы MD5 и распаковкой
в `data/raw/aurora/`.

## Проверки

```bash
pytest
```

## Прогноз по первым K циклов (инженерный сценарий)

```bash
python scripts/predict_cell.py --cell empa__ccid000208 \
    --history 50 --horizon 800 --levels 0.95 0.9 0.85 0.8 \
    --save-curves --out report.json
```

Конвейер: предобработка → идентификация состояния по префиксу →
калибровка кинетики деградации → рекурсивная симуляция будущих циклов
(шаблон тока повторяет последний наблюдённый цикл; свой профиль можно
передать через API `forecast`) → циклы пересечения порогов SOH и RUL.

## Конвейер обучения

```bash
python scripts/eda.py                  # EDA, контроль качества, сплит
python scripts/export_params.py        # литературные кривые (этап 1б)
python scripts/pretrain_ocv.py         # этап 0: предобучение OCV-сетей
python scripts/stage1_all.py --shard 0/4   # этап 1: идентификация (по шардам)
python scripts/stage2_train.py         # этап 2: encoder + базовые модели
python scripts/forecast_soh.py --cells <id...> --history 5 10 15 25 50 100
python scripts/forecast_report.py      # сводная таблица прогнозов
```

Артефакты: `checkpoints/stage{0,1,2}/`, `reports/forecast/`,
`reports/stage2_metrics.json`, `LAB_JOURNAL.md` — ход исследования.
