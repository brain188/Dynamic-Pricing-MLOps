"""Unit tests for ``dynamic_pricing.pipelines.feature_engineering.nodes``."""

import pandas as pd
import pytest

from dynamic_pricing.pipelines.feature_engineering.nodes import (
    assemble_feature_table,
    encode_categorical_features,
    engineer_demand_supply_ratio,
    fit_categorical_encoder,
)


@pytest.fixture
def base_parameters():
    """Returns a minimal, valid parameters dict mirroring parameters.yml."""
    return {
        "demand_supply_ratio": {
            "numerator_column": "Number_of_Riders",
            "denominator_column": "Number_of_Drivers",
            "outlier_cap_percentile": 0.9,
        },
        "categorical_columns": ["Location_Category", "Vehicle_Type"],
        "encoder": {"handle_unknown": "ignore", "drop": "first"},
        "target_column": "Historical_Cost_of_Ride",
    }


@pytest.fixture
def sample_data():
    """A small dataset with a range of demand-supply ratios, including an
    outlier, so the capping behavior is directly observable."""
    return pd.DataFrame(
        {
            "Number_of_Riders": [10, 20, 30, 40, 100],
            "Number_of_Drivers": [10, 10, 10, 10, 2],  # last row -> ratio of 50 (outlier)
            "Location_Category": ["Urban", "Rural", "Urban", "Suburban", "Rural"],
            "Vehicle_Type": ["Premium", "Economy", "Economy", "Premium", "Economy"],
            "Historical_Cost_of_Ride": [100.0, 150.0, 200.0, 250.0, 300.0],
        }
    )


class TestEngineerDemandSupplyRatio:
    def test_computes_ratio_correctly(self, sample_data, base_parameters):
        result = engineer_demand_supply_ratio(sample_data, base_parameters)
        assert "demand_supply_ratio" in result.columns
        assert result.loc[0, "demand_supply_ratio"] == pytest.approx(1.0)

    def test_caps_outlier_above_percentile(self, sample_data, base_parameters):
        result = engineer_demand_supply_ratio(sample_data, base_parameters)
        cap_value = (sample_data["Number_of_Riders"] / sample_data["Number_of_Drivers"]).quantile(0.9)
        assert result["demand_supply_ratio"].max() == pytest.approx(cap_value)
        assert result.loc[4, "demand_supply_ratio"] < 50.0  # the raw outlier ratio

    def test_raises_on_zero_denominator(self, sample_data, base_parameters):
        sample_data.loc[0, "Number_of_Drivers"] = 0
        with pytest.raises(ValueError, match="zero value"):
            engineer_demand_supply_ratio(sample_data, base_parameters)

    def test_preserves_original_columns(self, sample_data, base_parameters):
        result = engineer_demand_supply_ratio(sample_data, base_parameters)
        for column in sample_data.columns:
            assert column in result.columns


class TestFitCategoricalEncoder:
    def test_fits_without_error(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        assert encoder is not None

    def test_produces_expected_number_of_output_columns(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        # 3 Location_Category levels - 1 (drop='first') + 2 Vehicle_Type levels - 1 = 3
        output_columns = encoder.get_feature_names_out(base_parameters["categorical_columns"])
        assert len(output_columns) == 3


class TestEncodeCategoricalFeatures:
    def test_replaces_categorical_columns_with_encoded_columns(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        result = encode_categorical_features(sample_data, encoder, base_parameters)

        assert "Location_Category" not in result.columns
        assert "Vehicle_Type" not in result.columns
        assert any(col.startswith("Location_Category_") for col in result.columns)

    def test_passthrough_columns_unchanged(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        result = encode_categorical_features(sample_data, encoder, base_parameters)
        pd.testing.assert_series_equal(
            result["Historical_Cost_of_Ride"], sample_data["Historical_Cost_of_Ride"]
        )

    def test_handles_unseen_category_gracefully(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        new_data = sample_data.copy()
        new_data.loc[0, "Location_Category"] = "Unseen_Category"
        # should not raise, thanks to handle_unknown="ignore"
        result = encode_categorical_features(new_data, encoder, base_parameters)
        assert len(result) == len(new_data)


class TestAssembleFeatureTable:
    def test_target_column_is_last(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        encoded = encode_categorical_features(sample_data, encoder, base_parameters)
        result = assemble_feature_table(encoded, base_parameters)
        assert result.columns[-1] == "Historical_Cost_of_Ride"

    def test_raises_on_unexpected_nulls(self, sample_data, base_parameters):
        encoder = fit_categorical_encoder(sample_data, base_parameters)
        encoded = encode_categorical_features(sample_data, encoder, base_parameters)
        encoded.loc[0, "Historical_Cost_of_Ride"] = None
        with pytest.raises(ValueError, match="unexpected null"):
            assemble_feature_table(encoded, base_parameters)
