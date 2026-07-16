"""Unit tests for ``dynamic_pricing.pipelines.data_processing.nodes``.

Each node is tested in isolation with small, hand-built dataframes rather
than the real dataset, so tests stay fast and each failure points
precisely at the broken behavior rather than requiring the full 1000-row
historical dataset to reproduce.
"""

import pandas as pd
import pytest

from dynamic_pricing.pipelines.data_processing.nodes import (
    enforce_final_dtypes,
    handle_missing_values,
    remove_duplicate_rows,
    validate_schema,
    validate_value_ranges,
)


@pytest.fixture
def base_parameters():
    """Returns a minimal, valid parameters dict mirroring parameters.yml."""
    return {
        "expected_schema": {
            "Number_of_Riders": "int64",
            "Number_of_Drivers": "int64",
            "Average_Ratings": "float64",
        },
        "duplicate_subset": None,
        "missing_value_strategy": {
            "Average_Ratings": "median",
        },
        "value_ranges": {
            "Average_Ratings": {"min": 1.0, "max": 5.0},
        },
        "final_dtypes": {
            "Number_of_Riders": "int64",
            "Number_of_Drivers": "int64",
            "Average_Ratings": "float64",
        },
    }


class TestValidateSchema:
    def test_passes_with_matching_schema(self, base_parameters):
        df = pd.DataFrame(
            {
                "Number_of_Riders": pd.array([1, 2], dtype="int64"),
                "Number_of_Drivers": pd.array([1, 2], dtype="int64"),
                "Average_Ratings": pd.array([4.5, 3.2], dtype="float64"),
            }
        )
        result = validate_schema(df, base_parameters)
        pd.testing.assert_frame_equal(result, df)

    def test_raises_on_missing_column(self, base_parameters):
        df = pd.DataFrame({"Number_of_Riders": pd.array([1], dtype="int64")})
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_schema(df, base_parameters)

    def test_raises_on_dtype_mismatch(self, base_parameters):
        df = pd.DataFrame(
            {
                "Number_of_Riders": pd.array([1.0], dtype="float64"),  # wrong dtype
                "Number_of_Drivers": pd.array([1], dtype="int64"),
                "Average_Ratings": pd.array([4.5], dtype="float64"),
            }
        )
        with pytest.raises(ValueError, match="dtype mismatches"):
            validate_schema(df, base_parameters)


class TestRemoveDuplicateRows:
    def test_removes_exact_duplicates(self, base_parameters):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        result = remove_duplicate_rows(df, base_parameters)
        assert len(result) == 2

    def test_keeps_unique_rows_untouched(self, base_parameters):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        result = remove_duplicate_rows(df, base_parameters)
        assert len(result) == 3


class TestHandleMissingValues:
    def test_no_missing_values_returns_unchanged(self, base_parameters):
        df = pd.DataFrame({"Average_Ratings": [4.0, 5.0]})
        result = handle_missing_values(df, base_parameters)
        pd.testing.assert_frame_equal(result, df)

    def test_median_strategy_fills_value(self, base_parameters):
        df = pd.DataFrame({"Average_Ratings": [4.0, None, 6.0]})
        result = handle_missing_values(df, base_parameters)
        assert result["Average_Ratings"].isnull().sum() == 0
        assert result.loc[1, "Average_Ratings"] == 5.0  # median of 4.0 and 6.0

    def test_drop_row_strategy_for_unconfigured_column(self, base_parameters):
        df = pd.DataFrame(
            {
                "Average_Ratings": [4.0, 5.0],
                "Unconfigured_Column": [1.0, None],
            }
        )
        result = handle_missing_values(df, base_parameters)
        assert len(result) == 1

    def test_raises_on_unsupported_strategy(self, base_parameters):
        params = {**base_parameters, "missing_value_strategy": {"Average_Ratings": "bogus"}}
        df = pd.DataFrame({"Average_Ratings": [4.0, None]})
        with pytest.raises(ValueError, match="Unsupported missing_value_strategy"):
            handle_missing_values(df, params)


class TestValidateValueRanges:
    def test_removes_out_of_range_rows(self, base_parameters):
        df = pd.DataFrame({"Average_Ratings": [4.5, 0.5, 6.0, 3.0]})
        clean_df, report = validate_value_ranges(df, base_parameters)
        assert len(clean_df) == 2
        assert report.loc[0, "invalid_row_count"] == 2

    def test_no_bounds_configured_leaves_data_untouched(self):
        df = pd.DataFrame({"Average_Ratings": [4.5, 0.5]})
        clean_df, report = validate_value_ranges(df, {"value_ranges": {}})
        assert len(clean_df) == 2
        assert report.empty


class TestEnforceFinalDtypes:
    def test_casts_columns_correctly(self, base_parameters):
        df = pd.DataFrame(
            {
                "Number_of_Riders": [1.0, 2.0],
                "Number_of_Drivers": [1.0, 2.0],
                "Average_Ratings": [4, 5],
            }
        )
        result = enforce_final_dtypes(df, base_parameters)
        assert str(result["Number_of_Riders"].dtype) == "int64"
        assert str(result["Average_Ratings"].dtype) == "float64"

    def test_raises_on_uncastable_value(self, base_parameters):
        df = pd.DataFrame(
            {
                "Number_of_Riders": ["not_a_number"],
                "Number_of_Drivers": [1],
                "Average_Ratings": [4.0],
            }
        )
        with pytest.raises(ValueError):
            enforce_final_dtypes(df, base_parameters)
