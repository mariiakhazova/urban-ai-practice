"""
Расчётный движок: обеспеченность школами, дефицит, сценарии размещения.

Использует competitive_provision из blocksnet — тот же метод,
что в практике 2 (LP-оптимизация конкурентного распределения спроса).
"""

import math
from copy import deepcopy

import numpy as np
import pandas as pd
from blocksnet.analysis.provision import competitive_provision

SCHOOL_DEMAND = 120
SCHOOL_ACCESSIBILITY = 15


def compute_provision(
    population: pd.Series,
    capacity: pd.Series,
    acc_mx: pd.DataFrame,
    accessibility: int = SCHOOL_ACCESSIBILITY,
    demand_norm: int = SCHOOL_DEMAND,
) -> pd.DataFrame:
    prov_input = pd.DataFrame(
        {"population": population.fillna(0), "capacity": capacity.fillna(0)}
    )
    prov_input.index = population.index
    prov_df, _ = competitive_provision(
        prov_input, acc_mx, accessibility, demand_norm
    )
    return prov_df


def compute_scenario(
    base_population: pd.Series,
    base_capacity: pd.Series,
    acc_mx: pd.DataFrame,
    additions: dict,
    accessibility: int = SCHOOL_ACCESSIBILITY,
    demand_norm: int = SCHOOL_DEMAND,
) -> pd.DataFrame:
    new_capacity = base_capacity.copy()
    for block_id, cap_add in additions.items():
        new_capacity.loc[block_id] = new_capacity.loc[block_id] + cap_add
    return compute_provision(
        base_population, new_capacity, acc_mx, accessibility, demand_norm
    )


def total_deficit(prov_df: pd.DataFrame) -> float:
    return float(prov_df["demand_left"].sum())


def total_demand(prov_df: pd.DataFrame) -> float:
    return float(prov_df["demand"].sum())


def total_capacity(prov_df: pd.DataFrame) -> float:
    return float(prov_df["capacity"].sum())


def mean_provision_strong(prov_df: pd.DataFrame) -> float:
    populated = prov_df[prov_df["demand"] > 0]
    if len(populated) == 0:
        return 0.0
    return float(populated["provision_strong"].mean())


def mean_provision_weak(prov_df: pd.DataFrame) -> float:
    populated = prov_df[prov_df["demand"] > 0]
    if len(populated) == 0:
        return 0.0
    return float(populated["provision_weak"].mean())


def deficit_reduction(base_deficit: float, scenario_deficit: float) -> float:
    return base_deficit - scenario_deficit


def capacity_increase(base_capacity: pd.Series, additions: dict) -> int:
    return sum(additions.values())


def scenario_summary(
    prov_df: pd.DataFrame,
    base_deficit: float,
    base_capacity_total: float,
    additions: dict,
) -> dict:
    scen_deficit = total_deficit(prov_df)
    scen_demand = total_demand(prov_df)
    scen_capacity = total_capacity(prov_df)
    cap_inc = capacity_increase(base_capacity_total, additions)
    def_red = deficit_reduction(base_deficit, scen_deficit)
    return {
        "total_demand": scen_demand,
        "total_capacity": scen_capacity,
        "capacity_increase": cap_inc,
        "total_deficit": scen_deficit,
        "base_deficit": base_deficit,
        "deficit_reduction": def_red,
        "deficit_reduction_pct": (def_red / base_deficit * 100) if base_deficit > 0 else 0,
        "mean_provision_strong": mean_provision_strong(prov_df),
        "mean_provision_weak": mean_provision_weak(prov_df),
    }


def block_details(
    prov_df: pd.DataFrame, population: pd.Series, base_capacity: pd.Series
) -> pd.DataFrame:
    details = pd.DataFrame(index=prov_df.index)
    details["population"] = population
    details["base_capacity"] = base_capacity
    details["demand"] = prov_df["demand"]
    details["capacity"] = prov_df["capacity"]
    details["demand_left"] = prov_df["demand_left"]
    details["provision_strong"] = prov_df["provision_strong"]
    details["provision_weak"] = prov_df["provision_weak"]
    details["capacity_added"] = prov_df["capacity"] - base_capacity
    details["capacity_added"] = details["capacity_added"].clip(lower=0).astype(int)
    return details
