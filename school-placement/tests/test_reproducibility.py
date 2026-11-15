"""Tests for reproducibility: verify that our calculations match the original practice 2 results."""

import numpy as np
import pandas as pd
import pytest

import engine
import data_loader


def test_base_provision_matches_saved():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    computed_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    saved_demand = blocks["demand"].fillna(0)
    saved_capacity = blocks["capacity"].fillna(0)
    saved_prov_strong = blocks["provision_strong"].fillna(0)
    saved_prov_weak = blocks["provision_weak"].fillna(0)

    assert np.allclose(computed_prov["demand"], saved_demand), "Demand mismatch"
    assert np.allclose(computed_prov["capacity"], saved_capacity), "Capacity mismatch"

    populated = computed_prov["demand"] > 0
    assert np.allclose(
        computed_prov.loc[populated, "provision_strong"],
        saved_prov_strong[populated],
        atol=1e-6,
    ), "provision_strong mismatch"

    assert np.allclose(
        computed_prov.loc[populated, "provision_weak"],
        saved_prov_weak[populated],
        atol=1e-6,
    ), "provision_weak mismatch"


def test_total_demand_matches_saved():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    computed_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    saved_total_demand = blocks["demand"].sum()
    assert abs(computed_prov["demand"].sum() - saved_total_demand) < 1


def test_total_capacity_matches_saved():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    computed_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    saved_total_capacity = blocks["capacity"].sum()
    assert abs(computed_prov["capacity"].sum() - saved_total_capacity) < 1


def test_total_deficit_matches_saved():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    computed_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    saved_deficit = (blocks["demand"] - blocks["capacity"]).clip(lower=0).sum()
    computed_deficit = engine.total_deficit(computed_prov)

    assert computed_deficit > 0
    assert abs(computed_deficit - 6199) < 10, f"Expected deficit ~6199, got {computed_deficit}"


def test_no_schools_gives_zero_provision():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()

    zero_capacity = pd.Series(0, index=blocks.index, dtype=float)
    prov_df = engine.compute_provision(blocks["population"], zero_capacity, acc_mx)

    populated = prov_df[prov_df["demand"] > 0]
    assert (populated["provision_strong"] == 0).all()
    assert (populated["provision_weak"] == 0).all()


def test_scenario_with_additions_increases_capacity():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    base_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    base_capacity_total = base_prov["capacity"].sum()

    additions = {0: 600}
    scen_prov = engine.compute_scenario(
        prov_input["population"], prov_input["capacity"], acc_mx, additions
    )
    scen_capacity_total = scen_prov["capacity"].sum()

    assert scen_capacity_total == base_capacity_total + 600


def test_scenario_reduces_deficit():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    base_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    base_deficit = engine.total_deficit(base_prov)

    populated_blocks = blocks[blocks["population"] > 0]
    best_block = populated_blocks.index[0]

    additions = {best_block: 600}
    scen_prov = engine.compute_scenario(
        prov_input["population"], prov_input["capacity"], acc_mx, additions
    )
    scen_deficit = engine.total_deficit(scen_prov)

    assert scen_deficit <= base_deficit


def test_adding_school_to_populated_block_improves_provision():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    base_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    populated_blocks = blocks[blocks["population"] > 0]
    target_block = populated_blocks.index[0]

    additions = {target_block: 600}
    scen_prov = engine.compute_scenario(
        prov_input["population"], prov_input["capacity"], acc_mx, additions
    )

    assert scen_prov.loc[target_block, "capacity"] > base_prov.loc[target_block, "capacity"]


def test_scenario_does_not_modify_base_data():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    original_cap = prov_input["capacity"].copy()
    original_pop = prov_input["population"].copy()

    additions = {0: 600}
    engine.compute_scenario(
        prov_input["population"], prov_input["capacity"], acc_mx, additions
    )

    assert prov_input["capacity"].equals(original_cap)
    assert prov_input["population"].equals(original_pop)


def test_demand_calculation_matches_blocksnet():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    computed_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    import math

    expected_demand = prov_input["population"].apply(
        lambda p: math.ceil(p / 1000 * 120)
    )
    assert (computed_prov["demand"] == expected_demand).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
