"""Tests for data_loader module."""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from data_loader import (
    SCHOOL_ACCESSIBILITY,
    SCHOOL_DEMAND,
    load_accessibility_matrix,
    load_blocks,
    load_boundary,
    load_provision_input,
    load_schools,
    validate_data,
)


def test_school_constants():
    assert SCHOOL_DEMAND == 120
    assert SCHOOL_ACCESSIBILITY == 15


def test_load_blocks_shape():
    blocks = load_blocks()
    assert isinstance(blocks, gpd.GeoDataFrame)
    assert len(blocks) > 0
    assert "population" in blocks.columns
    assert "capacity" in blocks.columns
    assert "demand" in blocks.columns
    assert "provision_strong" in blocks.columns
    assert "provision_weak" in blocks.columns
    assert "geometry" in blocks.columns


def test_load_blocks_index():
    blocks = load_blocks()
    assert len(blocks.index.unique()) == len(blocks)
    assert blocks.index.name is None


def test_load_blocks_population_non_negative():
    blocks = load_blocks()
    assert (blocks["population"] >= 0).all()


def test_load_blocks_capacity_non_negative():
    blocks = load_blocks()
    assert (blocks["capacity"] >= 0).all()


def test_load_accessibility_matrix_shape():
    acc_mx = load_accessibility_matrix()
    assert isinstance(acc_mx, pd.DataFrame)
    assert acc_mx.shape[0] == acc_mx.shape[1]
    assert acc_mx.shape[0] > 0


def test_load_accessibility_matrix_index_match_columns():
    acc_mx = load_accessibility_matrix()
    assert list(acc_mx.index) == list(acc_mx.columns)


def test_load_accessibility_matrix_index_type():
    acc_mx = load_accessibility_matrix()
    assert acc_mx.index.dtype in [np.int64, int]
    assert acc_mx.columns.dtype in [np.int64, int]


def test_load_accessibility_matrix_no_nan():
    acc_mx = load_accessibility_matrix()
    assert not acc_mx.isna().any().any()


def test_load_accessibility_matrix_diagonal_zero():
    acc_mx = load_accessibility_matrix()
    np.testing.assert_array_almost_equal(np.diag(acc_mx.values), 0, decimal=2)


def test_load_accessibility_matrix_non_negative():
    acc_mx = load_accessibility_matrix()
    assert (acc_mx.values >= 0).all()


def test_load_accessibility_matrix_approximately_symmetric():
    acc_mx = load_accessibility_matrix()
    diff = np.abs(acc_mx.values - acc_mx.values.T)
    max_diff = diff.max()
    assert max_diff < 2.0, f"Matrix not approximately symmetric, max diff={max_diff}"


def test_load_boundary():
    boundary = load_boundary()
    assert isinstance(boundary, gpd.GeoDataFrame)
    assert len(boundary) > 0
    assert "geometry" in boundary.columns


def test_load_schools():
    schools = load_schools()
    assert isinstance(schools, gpd.GeoDataFrame)
    assert len(schools) > 0
    assert "capacity" in schools.columns
    assert "geometry" in schools.columns


def test_load_schools_capacity_positive():
    schools = load_schools()
    assert (schools["capacity"] > 0).all()


def test_load_provision_input():
    blocks = load_blocks()
    prov_input = load_provision_input(blocks)
    assert isinstance(prov_input, pd.DataFrame)
    assert "population" in prov_input.columns
    assert "capacity" in prov_input.columns
    assert len(prov_input) == len(blocks)
    assert prov_input.index.equals(blocks.index)
    assert not prov_input["population"].isna().any()
    assert not prov_input["capacity"].isna().any()


def test_validate_data_ok():
    blocks = load_blocks()
    acc_mx = load_accessibility_matrix()
    report = validate_data(blocks, acc_mx)
    assert report["ok"] is True
    assert "summary" in report
    assert "n_blocks" in report["summary"]
    assert report["summary"]["n_blocks"] == len(blocks)


def test_validate_data_matrix_matches_blocks():
    blocks = load_blocks()
    acc_mx = load_accessibility_matrix()
    report = validate_data(blocks, acc_mx)
    assert report["ok"] is True
    assert report["summary"]["n_blocks"] == acc_mx.shape[0]


def test_validate_data_summary():
    blocks = load_blocks()
    acc_mx = load_accessibility_matrix()
    report = validate_data(blocks, acc_mx)
    summ = report["summary"]
    assert summ["n_blocks"] > 0
    assert summ["n_populated"] > 0
    assert summ["total_population"] > 0
    assert summ["total_capacity"] > 0
    assert summ["mx_mean"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
