"""Integration smoke test: trains and registers a real (tiny) CatBoost
model against a real, temporary MLflow SQLite store — mirroring what
`training` actually produced (CatBoost won) — fits a real
`categorical_encoder`, and then runs the ENTIRE `inference` chain against
it: schema validation, demand-supply ratio engineering, categorical
encoding, model loading by alias, prediction, and sanity checking.

This is the only way to actually verify the training/serving contract
holds: that a model trained on features produced by
`feature_engineering`'s functions can be fed, at inference time, features
produced by those same functions reused unchanged.
"""

import mlflow
import mlflow.catboost
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from mlflow.tracking import MlflowClient

from dynamic_pricing.pipelines.data_processing.nodes import validate_schema
from dynamic_pricing.pipelines.feature_engineering.nodes import (
    encode_categorical_features,
    engineer_demand_supply_ratio,
    fit_categorical_encoder,
)
from dynamic_pricing.pipelines.inference.nodes import (
    generate_predictions,
    load_production_model,
    validate_predictions,
)


@pytest.fixture
def feature_engineering_parameters():
    return {
        "demand_supply_ratio": {
            "numerator_column": "Number_of_Riders",
            "denominator_column": "Number_of_Drivers",
            "outlier_cap_percentile": 0.99,
        },
        "categorical_columns": ["Location_Category", "Vehicle_Type"],
        "encoder": {"handle_unknown": "ignore", "drop": "first"},
        "target_column": "Historical_Cost_of_Ride",
    }


@pytest.fixture
def inference_parameters(tmp_path):
    db_path = tmp_path / "mlflow_test.db"
    return {
        "expected_schema": {
            "Number_of_Riders": "int64",
            "Number_of_Drivers": "int64",
            "Location_Category": "object",
            "Vehicle_Type": "object",
            "Expected_Ride_Duration": "int64",
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{db_path}",
            "registered_model_name": "test_dynamic_pricing_model",
            "production_alias": "production",
        },
        "prediction_column_name": "predicted_fare",
        "sanity_bounds": {"min": 0, "max": 5000, "action": "clip"},
    }


@pytest.fixture
def historical_training_data():
    """Mirrors the shape of the real historical dataset closely enough to
    fit a real encoder and train a real (tiny) model on it."""
    return pd.DataFrame(
        {
            "Number_of_Riders": [90, 58, 42, 89, 70, 33, 12, 95, 60, 45],
            "Number_of_Drivers": [45, 39, 31, 28, 33, 20, 11, 40, 30, 22],
            "Location_Category": [
                "Urban", "Suburban", "Rural", "Rural", "Urban",
                "Suburban", "Rural", "Urban", "Suburban", "Rural",
            ],
            "Vehicle_Type": [
                "Premium", "Economy", "Premium", "Economy", "Premium",
                "Economy", "Premium", "Economy", "Premium", "Economy",
            ],
            "Expected_Ride_Duration": [90, 60, 45, 100, 80, 30, 150, 70, 55, 65],
            "Historical_Cost_of_Ride": [300.0, 250.0, 220.0, 400.0, 310.0, 150.0, 480.0, 260.0, 240.0, 230.0],
        }
    )


def test_full_inference_chain_runs_against_a_really_trained_and_registered_model(
    historical_training_data, feature_engineering_parameters, inference_parameters
):
    # --- Arrange: reproduce what `feature_engineering` + `training` already did ---
    data_with_ratio = engineer_demand_supply_ratio(
        historical_training_data, feature_engineering_parameters
    )
    encoder = fit_categorical_encoder(data_with_ratio, feature_engineering_parameters)
    encoded_data = encode_categorical_features(data_with_ratio, encoder, feature_engineering_parameters)

    X_train = encoded_data.drop(columns=["Historical_Cost_of_Ride"])
    y_train = encoded_data["Historical_Cost_of_Ride"]

    model = CatBoostRegressor(iterations=20, depth=3, verbose=False, random_seed=42)
    model.fit(X_train, y_train)

    mlflow.set_tracking_uri(inference_parameters["mlflow"]["tracking_uri"])
    mlflow.set_experiment("test_inference_integration")
    with mlflow.start_run():
        info = mlflow.catboost.log_model(
            model,
            name="model",
            registered_model_name=inference_parameters["mlflow"]["registered_model_name"],
            input_example=X_train.head(3),
        )

    client = MlflowClient(tracking_uri=inference_parameters["mlflow"]["tracking_uri"])
    versions = client.search_model_versions(
        f"name='{inference_parameters['mlflow']['registered_model_name']}'"
    )
    client.set_registered_model_alias(
        inference_parameters["mlflow"]["registered_model_name"], "production", versions[0].version
    )

    # --- Act: run the full inference chain, exactly as pipeline.py wires it ---
    new_ride_requests = pd.DataFrame(
        {
            "Number_of_Riders": [75, 40],
            "Number_of_Drivers": [35, 25],
            "Location_Category": ["Urban", "Rural"],
            "Customer_Loyalty_Status": ["Gold", "Silver"],  # extra col, not in expected_schema on purpose
            "Number_of_Past_Rides": [20, 10],
            "Average_Ratings": [4.5, 4.0],
            "Time_of_Booking": ["Night", "Morning"],
            "Vehicle_Type": ["Premium", "Economy"],
            "Expected_Ride_Duration": [95, 50],
        }
    )
    # Trim to only the columns declared in expected_schema, mirroring what a
    # real inference request payload would look like.
    new_ride_requests = new_ride_requests[list(inference_parameters["expected_schema"].keys())]

    validated = validate_schema(new_ride_requests, inference_parameters)
    with_ratio = engineer_demand_supply_ratio(validated, feature_engineering_parameters)
    encoded = encode_categorical_features(with_ratio, encoder, feature_engineering_parameters)

    production_model = load_production_model(inference_parameters)
    raw_predictions = generate_predictions(encoded, production_model, inference_parameters)
    final_predictions = validate_predictions(raw_predictions, inference_parameters)

    # --- Assert ---
    assert "predicted_fare" in final_predictions.columns
    assert len(final_predictions) == 2
    assert final_predictions["predicted_fare"].notnull().all()
    assert (final_predictions["predicted_fare"] >= 0).all()
    assert "is_out_of_bounds" in final_predictions.columns

    print("Full inference chain executed successfully against a real registered model.")
    print(final_predictions[["predicted_fare", "is_out_of_bounds"]])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
