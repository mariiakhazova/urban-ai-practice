"""Tests for engine module (calculation engine)."""

import numpy as np
import pandas as pd
import pytest

import engine
import data_loader


def test_compute_provision_returns_dataframe():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_df = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    assert isinstance(prov_df, pd.DataFrame)
    assert len(prov_df) == len(blocks)


def test_compute_provision_columns():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_df = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    expected_cols = [
        "demand",
        "capacity",
        "demand_left",
        "demand_within",
        "demand_without",
        "capacity_left",
        "capacity_within",
        "capacity_without",
        "provision_strong",
        "provision_weak",
    ]
    for col in expected_cols:
        assert col in prov_df.columns


def test_compute_provision_demand_positive():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_df = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    assert (prov_df["demand"] >= 0).all()


def test_compute_provision_capacity_positive():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_df = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    assert (prov_df["capacity"] >= 0).all()


def test_compute_provision_demand_left_non_negative():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_df = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    assert (prov_df["demand_left"] >= 0).all()


def test_compute_provision_provision_in_range():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_df = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    populated = prov_df[prov_df["demand"] > 0]
    assert (populated["provision_strong"] >= 0).all()
    assert (populated["provision_strong"] <= 1).all()
    assert (populated["provision_weak"] >= 0).all()
    assert (populated["provision_weak"] <= 1).all()


def test_total_deficit():
    prov_df = pd.DataFrame(
        {"demand": [10, 20, 30], "demand_left": [5, 10, 15]}
    )
    assert engine.total_deficit(prov_df) == 30.0


def test_total_demand():
    prov_df = pd.DataFrame({"demand": [10, 20, 30]})
    assert engine.total_demand(prov_df) == 60.0


def test_total_capacity():
    prov_df = pd.DataFrame({"capacity": [5, 10, 15]})
    assert engine.total_capacity(prov_df) == 30.0


def test_mean_provision_strong():
    prov_df = pd.DataFrame(
        {"demand": [10, 20, 30], "provision_strong": [0.5, 0.75, 1.0]}
    )
    assert engine.mean_provision_strong(prov_df) == pytest.approx(0.75)


def test_mean_provision_weak():
    prov_df = pd.DataFrame(
        {"demand": [10, 20, 30], "provision_weak": [0.5, 0.75, 1.0]}
    )
    assert engine.mean_provision_weak(prov_df) == pytest.approx(0.75)


def test_deficit_reduction():
    base_deficit = 100.0
    scenario_deficit = 70.0
    assert engine.deficit_reduction(base_deficit, scenario_deficit) == 30.0


def test_capacity_increase():
    base_cap = pd.Series([10, 20, 30], index=[0, 1, 2])
    additions = {0: 5, 2: 10}
    assert engine.capacity_increase(base_cap, additions) == 15


def test_compute_scenario_add_capacity():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    base_prov = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    base_capacity = base_prov["capacity"].sum()

    additions = {0: 100}
    scen_prov = engine.compute_scenario(
        prov_input["population"],
        prov_input["capacity"],
        acc_mx,
        additions,
    )
    scen_capacity = scen_prov["capacity"].sum()

    assert scen_capacity == base_capacity + 100


def test_compute_scenario_doesnt_modify_original():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)
    prov_input_copy = prov_input.copy()

    additions = {0: 100}
    scen_prov = engine.compute_scenario(
        prov_input["population"],
        prov_input["capacity"],
        acc_mx,
        additions,
    )

    assert prov_input["capacity"].equals(prov_input_copy["capacity"])


def test_scenario_summary():
    prov_df = pd.DataFrame(
        {
            "demand": [100, 200, 300],
            "capacity": [50, 100, 150],
            "demand_left": [50, 100, 150],
            "provision_strong": [0.5, 0.5, 0.5],
            "provision_weak": [0.5, 0.5, 0.5],
        }
    )
    base_deficit = 600.0
    base_cap_total = 300.0
    additions = {0: 50, 1: 50}

    summary = engine.scenario_summary(
        prov_df, base_deficit, base_cap_total, additions
    )

    assert summary["total_demand"] == 600.0
    assert summary["total_capacity"] == 300.0
    assert summary["capacity_increase"] == 100.0
    assert summary["total_deficit"] == 300.0
    assert summary["base_deficit"] == 600.0
    assert summary["deficit_reduction"] == 300.0
    assert summary["deficit_reduction_pct"] == pytest.approx(50.0)
    assert summary["mean_provision_strong"] == pytest.approx(0.5)
    assert summary["mean_provision_weak"] == pytest.approx(0.5)


def test_block_details():
    prov_df = pd.DataFrame(
        {
            "demand": [10, 20, 30],
            "capacity": [5, 10, 15],
            "demand_left": [5, 10, 15],
            "provision_strong": [0.5, 0.5, 0.5],
            "provision_weak": [0.5, 0.5, 0.5],
        },
        index=[0, 1, 2],
    )
    population = pd.Series([100, 200, 300], index=[0, 1, 2])
    base_capacity = pd.Series([5, 10, 15], index=[0, 1, 2])

    details = engine.block_details(prov_df, population, base_capacity)

    assert isinstance(details, pd.DataFrame)
    assert len(details) == 3
    assert "population" in details.columns
    assert "base_capacity" in details.columns
    assert "demand" in details.columns
    assert "capacity" in details.columns
    assert "capacity_added" in details.columns


def test_reproducibility_of_base_provision():
    blocks = data_loader.load_blocks()
    acc_mx = data_loader.load_accessibility_matrix()
    prov_input = data_loader.load_provision_input(blocks)

    prov_df1 = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )
    prov_df2 = engine.compute_provision(
        prov_input["population"], prov_input["capacity"], acc_mx
    )

    assert (prov_df1["demand"] == prov_df2["demand"]).all()
    assert (prov_df1["capacity"] == prov_df2["capacity"]).all()
    assert (prov_df1["demand_left"] == prov_df2["demand_left"]).all()
    populated = prov_df1["demand"] > 0
    assert np.allclose(
        prov_df1.loc[populated, "provision_strong"],
        prov_df2.loc[populated, "provision_strong"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
