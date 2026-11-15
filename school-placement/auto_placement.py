"""
Алгоритм автоматического подбора квартала для размещения школы.

Перебирает все кварталы: для каждого независимо моделирует размещение
одной школы заданной вместимости и рассчитывает обеспеченность.
Выбирает квартал с максимальным сокращением суммарного дефицита.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from engine import compute_scenario, total_deficit


@dataclass
class PlacementOption:
    block_id: int
    capacity_added: int
    deficit: float
    deficit_reduction: float
    error: str | None = None


@dataclass
class AutoPlacementResult:
    options: list[PlacementOption] = field(default_factory=list)
    best_options: list[PlacementOption] = field(default_factory=list)
    n_checked: int = 0
    n_errors: int = 0
    n_total: int = 0
    incomplete: bool = False


def auto_placement(
    base_population: pd.Series,
    base_capacity: pd.Series,
    acc_mx: pd.DataFrame,
    school_capacity: int = 600,
    accessibility: int = 15,
    demand_norm: int = 120,
    progress_callback=None,
) -> AutoPlacementResult:
    result = AutoPlacementResult()
    all_block_ids = base_population.index.tolist()
    result.n_total = len(all_block_ids)

    base_deficit = total_deficit(
        compute_scenario(
            base_population,
            base_capacity,
            acc_mx,
            {},
            accessibility,
            demand_norm,
        )
    )

    options = []
    n_checked = 0
    n_errors = 0

    for i, block_id in enumerate(all_block_ids):
        try:
            additions = {block_id: school_capacity}
            prov_df = compute_scenario(
                base_population,
                base_capacity,
                acc_mx,
                additions,
                accessibility,
                demand_norm,
            )
            deficit = total_deficit(prov_df)
            reduction = base_deficit - deficit
            opt = PlacementOption(
                block_id=block_id,
                capacity_added=school_capacity,
                deficit=deficit,
                deficit_reduction=reduction,
            )
            options.append(opt)
        except Exception as e:
            opt = PlacementOption(
                block_id=block_id,
                capacity_added=school_capacity,
                deficit=base_deficit,
                deficit_reduction=0.0,
                error=str(e),
            )
            options.append(opt)
            n_errors += 1

        n_checked += 1
        if progress_callback is not None:
            progress_callback(n_checked, result.n_total, block_id, opt, result.n_errors)

    result.options = sorted(options, key=lambda o: o.deficit_reduction, reverse=True)
    result.n_checked = n_checked
    result.n_errors = n_errors

    if result.options:
        best_reduction = result.options[0].deficit_reduction
        tol = 1e-6
        result.best_options = [
            o
            for o in result.options
            if abs(o.deficit_reduction - best_reduction) < tol and o.error is None
        ]

    return result


def auto_placement_to_df(result: AutoPlacementResult) -> pd.DataFrame:
    rows = []
    for opt in result.options:
        rows.append(
            {
                "block_id": opt.block_id,
                "capacity_added": opt.capacity_added,
                "deficit": opt.deficit,
                "deficit_reduction": opt.deficit_reduction,
                "is_best": opt in result.best_options,
                "error": opt.error,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("deficit_reduction", ascending=False).reset_index(
            drop=True
        )
        df.index = df.index + 1
        df.index.name = "rank"
    return df
