"""Tests for auto_placement module using small synthetic data for speed."""

import math

import numpy as np
import pandas as pd
import pytest

import auto_placement
import engine


@pytest.fixture
def small_data():
    n = 10
    rng = np.random.RandomState(42)
    population = pd.Series(rng.randint(0, 5000, n), index=range(n), dtype=float)
    capacity = pd.Series(rng.randint(0, 200, n), index=range(n), dtype=float)
    acc_mx = pd.DataFrame(rng.uniform(1, 30, (n, n)), index=range(n), columns=range(n))
    np.fill_diagonal(acc_mx.values, 0)
    acc_mx = (acc_mx + acc_mx.T) / 2
    return population, capacity, acc_mx


def test_auto_placement_returns_result(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    assert hasattr(result, "options")
    assert hasattr(result, "best_options")
    assert hasattr(result, "n_checked")
    assert hasattr(result, "n_total")


def test_auto_placement_checks_all_blocks(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    assert result.n_checked == result.n_total
    assert result.n_checked == len(pop)


def test_auto_placement_options_count(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    assert len(result.options) == len(pop)


def test_auto_placement_best_options_exist(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    assert len(result.best_options) >= 1


def test_auto_placement_all_block_ids_covered(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    block_ids = {opt.block_id for opt in result.options}
    assert block_ids == set(pop.index)


def test_auto_placement_deficit_reduction_non_negative(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    for opt in result.options:
        assert opt.deficit_reduction >= 0


def test_auto_placement_best_has_highest_reduction(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    if result.best_options and result.options:
        best_reduction = result.best_options[0].deficit_reduction
        for opt in result.options:
            assert opt.deficit_reduction <= best_reduction + 1e-6


def test_auto_placement_manual_matches_auto(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    if result.best_options:
        best_block_id = result.best_options[0].block_id
        best_opt = result.best_options[0]
        manual_prov = engine.compute_scenario(pop, cap, mx, {best_block_id: 100})
        manual_deficit = engine.total_deficit(manual_prov)
        assert abs(manual_deficit - best_opt.deficit) < 1e-6


def test_auto_placement_options_sorted(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    reductions = [opt.deficit_reduction for opt in result.options]
    assert reductions == sorted(reductions, reverse=True)


def test_auto_placement_to_df(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    df = auto_placement.auto_placement_to_df(result)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(pop)
    assert "block_id" in df.columns
    assert "deficit_reduction" in df.columns
    assert "is_best" in df.columns


def test_auto_placement_to_df_sorted(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    df = auto_placement.auto_placement_to_df(result)
    reductions = df["deficit_reduction"].values
    assert all(reductions[i] >= reductions[i + 1] for i in range(len(reductions) - 1))


def test_auto_placement_doesnt_modify_input(small_data):
    pop, cap, mx = small_data
    orig_pop = pop.copy()
    orig_cap = cap.copy()
    auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    assert pop.equals(orig_pop)
    assert cap.equals(orig_cap)


def test_auto_placement_equal_best_options():
    n = 5
    population = pd.Series(1000.0, index=range(n))
    capacity = pd.Series(0.0, index=range(n))
    acc_mx = pd.DataFrame(5.0, index=range(n), columns=range(n))
    np.fill_diagonal(acc_mx.values, 0)
    result = auto_placement.auto_placement(population, capacity, acc_mx, school_capacity=100)
    assert len(result.best_options) == n, "All blocks should be equally good when symmetric"


def test_auto_placement_capacity_affects_results(small_data):
    pop, cap, mx = small_data
    result_50 = auto_placement.auto_placement(pop, cap, mx, school_capacity=50)
    result_200 = auto_placement.auto_placement(pop, cap, mx, school_capacity=200)
    opt_50 = result_50.options[0]
    opt_200 = result_200.options[0]
    assert opt_200.deficit_reduction >= opt_50.deficit_reduction


def test_auto_placement_no_errors(small_data):
    pop, cap, mx = small_data
    result = auto_placement.auto_placement(pop, cap, mx, school_capacity=100)
    assert result.n_errors == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
