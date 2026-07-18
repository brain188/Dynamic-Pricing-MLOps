"""End-to-end integration tests for the FastAPI backend.

Trains a real (tiny) CatBoost model, fits a real categorical encoder,
registers the model in a temporary MLflow SQLite store, aliases it
`"production"`, persists the encoder to disk, points the app's Settings
at all of it via environment variables, and then drives the actual app
through `TestClient` — no mocks. This is the only way to prove the
backend's reuse of `feature_engineering`/`inference` node functions
genuinely produces predictions consistent with how the model was trained.
"""

import os

import mlflow
import mlflow.catboost
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from dynamic_pricing.pipelines.feature_engineering.nodes import (
    encode_categorical_features,
    engineer_demand_supply_ratio,
    fit_categorical_encoder,
)
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient


FE_PARAMS = {
    "demand_supply_ratio": {
        "numerator_column": "Number_of_Riders",
        "denominator_column": "Number_of_Drivers",
        "outlier_cap_percentile": 0.99,
    },
    "categorical_columns": [
        "Location_Category",
        "Customer_Loyalty_Status",
        "Time_of_Booking",
        "Vehicle_Type",
    ],
    "encoder": {"handle_unknown": "ignore", "drop": "first"},
    "target_column": "Historical_Cost_of_Ride",
}


@pytest.fixture
def training_data():
    n = 40
    return pd.DataFrame(
        {
            "Number_of_Riders": [20 + i * 2 for i in range(n)],
            "Number_of_Drivers": [5 + i for i in range(n)],
            "Location_Category": (["Urban", "Suburban", "Rural"] * n)[:n],
            "Customer_Loyalty_Status": (["Silver", "Regular", "Gold"] * n)[:n],
            "Number_of_Past_Rides": [i for i in range(n)],
            "Average_Ratings": [3.5 + (i % 15) * 0.1 for i in range(n)],
            "Time_of_Booking": (["Morning", "Afternoon", "Evening", "Night"] * n)[:n],
            "Vehicle_Type": (["Economy", "Premium"] * n)[:n],
            "Expected_Ride_Duration": [10 + i * 4 for i in range(n)],
            "Historical_Cost_of_Ride": [50.0 + i * 15 for i in range(n)],
        }
    )


@pytest.fixture
def live_app(tmp_path, training_data, monkeypatch):
    """Trains + registers a real model, persists a real encoder, points
    the app at both via env vars, then yields a TestClient wired to the
    real (not mocked) app lifespan."""
    db_path = tmp_path / "mlflow_test.db"
    encoder_path = tmp_path / "categorical_encoder.pickle"
    audit_path = tmp_path / "audit.jsonl"
    tracking_uri = f"sqlite:///{db_path}"

    data_with_ratio = engineer_demand_supply_ratio(training_data, FE_PARAMS)
    encoder = fit_categorical_encoder(data_with_ratio, FE_PARAMS)
    encoded = encode_categorical_features(data_with_ratio, encoder, FE_PARAMS)

    import pickle

    with encoder_path.open("wb") as f:
        pickle.dump(encoder, f)

    X_train = encoded.drop(columns=["Historical_Cost_of_Ride"])
    y_train = encoded["Historical_Cost_of_Ride"]
    model = CatBoostRegressor(iterations=15, depth=3, verbose=False, random_seed=42)
    model.fit(X_train, y_train)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_backend_integration")
    with mlflow.start_run():
        info = mlflow.catboost.log_model(
            model,
            name="model",
            registered_model_name="test_backend_model",
            input_example=X_train.head(3),
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions("name='test_backend_model'")
    client.set_registered_model_alias("test_backend_model", "production", versions[0].version)

    monkeypatch.setenv("DYNAMIC_PRICING_MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("DYNAMIC_PRICING_REGISTERED_MODEL_NAME", "test_backend_model")
    monkeypatch.setenv("DYNAMIC_PRICING_ENCODER_PATH", str(encoder_path))
    monkeypatch.setenv("DYNAMIC_PRICING_AUDIT_LOG_PATH", str(audit_path))

    # Settings is cached via lru_cache; clear it so the env vars above
    # actually take effect for this test's app instance.
    from app.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as test_client:
        test_client.audit_path = audit_path  # stash for assertions below
        yield test_client

    get_settings.cache_clear()


VALID_REQUEST = {
    "Number_of_Riders": 90,
    "Number_of_Drivers": 45,
    "Location_Category": "Urban",
    "Customer_Loyalty_Status": "Silver",
    "Number_of_Past_Rides": 10,
    "Average_Ratings": 4.5,
    "Time_of_Booking": "Night",
    "Vehicle_Type": "Premium",
    "Expected_Ride_Duration": 90,
}


class TestHealth:
    def test_returns_ok_when_model_loaded(self, live_app):
        response = live_app.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] is not None


class TestPredict:
    def test_valid_request_returns_prediction(self, live_app):
        response = live_app.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_fare"] >= 0
        assert "request_id" in body
        assert body["model_version"] is not None
        assert response.headers["X-Request-ID"] == body["request_id"]

    def test_invalid_vehicle_type_returns_422(self, live_app):
        bad_request = {**VALID_REQUEST, "Vehicle_Type": "Helicopter"}
        response = live_app.post("/predict", json=bad_request)
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_zero_drivers_rejected_before_reaching_model(self, live_app):
        bad_request = {**VALID_REQUEST, "Number_of_Drivers": 0}
        response = live_app.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_successful_prediction_is_audited(self, live_app):
        live_app.post("/predict", json=VALID_REQUEST)
        audit_lines = live_app.audit_path.read_text().strip().split("\n")
        prediction_records = [
            line for line in audit_lines if '"event_type": "prediction"' in line
        ]
        assert len(prediction_records) >= 1
        assert '"status": "success"' in prediction_records[-1]

    def test_failed_request_is_audited_as_failure(self, live_app):
        bad_request = {**VALID_REQUEST, "Vehicle_Type": "Helicopter"}
        live_app.post("/predict", json=bad_request)
        audit_lines = live_app.audit_path.read_text().strip().split("\n")
        failure_records = [line for line in audit_lines if '"status": "failure"' in line]
        assert len(failure_records) >= 1
