"""
Модуль загрузки и подготовки данных из исходного проекта.

Загружает кварталы, население, вместимость школ, матрицу доступности
из сохранённых файлов практики 2 (Арзамас).
Все данные кэшируются для повторного запуска без скачивания.
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "arzamas"

SCHOOL_DEMAND = 120
SCHOOL_ACCESSIBILITY = 15


def _check_data_dir():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Папка с данными не найдена: {DATA_DIR}\n"
            "Убедитесь, что проект python-package-boilerplate находится рядом с school-placement."
        )


def load_blocks() -> gpd.GeoDataFrame:
    _check_data_dir()
    path = DATA_DIR / "blocks_metrics.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Файл кварталов не найден: {path}")
    blocks = gpd.read_file(path)
    blocks = blocks.set_index(blocks.index)
    for col in ["population", "capacity", "demand"]:
        if col in blocks.columns:
            blocks[col] = blocks[col].fillna(0).astype(float)
    for col in ["provision_strong", "provision_weak"]:
        if col in blocks.columns:
            blocks[col] = blocks[col].fillna(0.0)
    return blocks


def load_accessibility_matrix() -> pd.DataFrame:
    _check_data_dir()
    path = DATA_DIR / "acc_mx_drive.csv"
    if not path.exists():
        raise FileNotFoundError(f"Файл матрицы доступности не найден: {path}")
    acc_mx = pd.read_csv(path, index_col=0)
    acc_mx.columns = acc_mx.columns.astype(int)
    acc_mx.index = acc_mx.index.astype(int)
    return acc_mx


def load_boundary() -> gpd.GeoDataFrame:
    _check_data_dir()
    path = DATA_DIR / "boundary.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Файл границы города не найден: {path}")
    return gpd.read_file(path)


def load_schools() -> gpd.GeoDataFrame:
    _check_data_dir()
    path = DATA_DIR / "schools.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Файл школ не найден: {path}")
    return gpd.read_file(path)


def load_provision_input(blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    df = blocks[["population", "capacity"]].copy()
    df = df.fillna(0)
    return df


def validate_data(blocks: gpd.GeoDataFrame, acc_mx: pd.DataFrame) -> dict:
    report = {"errors": [], "warnings": [], "ok": True}

    n_blocks = len(blocks)
    n_mx = acc_mx.shape[0]

    if n_blocks != n_mx:
        report["errors"].append(
            f"Число кварталов ({n_blocks}) не совпадает с размером матрицы ({n_mx})"
        )
        report["ok"] = False

    blocks_ids = set(blocks.index)
    mx_ids = set(acc_mx.index)
    if blocks_ids != mx_ids:
        missing_in_mx = blocks_ids - mx_ids
        missing_in_blocks = mx_ids - blocks_ids
        if missing_in_mx:
            report["errors"].append(
                f"Кварталы без строк в матрице: {sorted(missing_in_mx)[:10]}"
            )
        if missing_in_blocks:
            report["errors"].append(
                f"Строки матрицы без кварталов: {sorted(missing_in_blocks)[:10]}"
            )
        report["ok"] = False

    mx_values = acc_mx.values
    n_nan = int(np.isnan(mx_values).sum())
    if n_nan > 0:
        report["errors"].append(f"Матрица содержит {n_nan} NaN-значений")
        report["ok"] = False

    n_neg = int((mx_values < 0).sum())
    if n_neg > 0:
        report["errors"].append(f"Матрица содержит {n_neg} отрицательных значений")
        report["ok"] = False

    diag = np.diag(mx_values)
    nonzero_diag = int((diag != 0).sum())
    if nonzero_diag > 0:
        report["warnings"].append(
            f"Диагональ матрицы: {nonzero_diag} ненулевых значений"
        )

    total_pop = blocks["population"].sum()
    total_cap = blocks["capacity"].sum()
    total_demand = blocks["demand"].sum() if "demand" in blocks.columns else 0
    n_populated = int((blocks["population"] > 0).sum())
    n_with_capacity = int((blocks["capacity"] > 0).sum())

    report["summary"] = {
        "n_blocks": n_blocks,
        "n_populated": n_populated,
        "n_with_capacity": n_with_capacity,
        "total_population": float(total_pop),
        "total_capacity": float(total_cap),
        "total_demand": float(total_demand),
        "mx_mean": float(mx_values.mean()),
        "mx_max": float(mx_values.max()),
    }

    return report
