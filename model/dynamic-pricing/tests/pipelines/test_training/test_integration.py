"""Integration smoke test: runs the full `training` node chain end-to-end,
including *real* MLflow experiment logging and model registration against
a temporary local SQLite tracking store (not a mock) — this is the only
way to actually verify the registration/alias workflow works, since
mocking MlflowClient would not catch a broken API call.
"""

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from dynamic_pricing.pipelines.training.nodes import (
    collect_trained_models,
    evaluate_models,
    log_and_register_best_model,
    split_data,
    train_baseline_model,
    train_catboost_model,
    train_lightgbm_model,
    train_random_forest,
    train_xgboost_model,
)


@pytest.fixture
def parameters(tmp_path):
    db_path = tmp_path / "mlflow_test.db"
    return {
        "target_column": "cost",
        "test_size": 0.25,
        "random_state": 42,
        "baseline_model_key": "linear_regression_baseline",
        "baseline_feature_columns": ["duration"],
        "tuning": {
            "cv_folds": 3,
            "n_iter": 3,
            "scoring": "neg_root_mean_squared_error",
            "random_state": 42,
        },
        "random_forest": {
            "fixed_params": {"random_state": 42, "n_jobs": 1},
            "search_space": {"n_estimators": [10, 20], "max_depth": [3, 4]},
        },
        "lightgbm": {
            "fixed_params": {"random_state": 42, "verbosity": -1},
            "search_space": {"n_estimators": [10, 20], "max_depth": [3, 4], "num_leaves": [7, 15]},
        },
        "xgboost": {
            "fixed_params": {"random_state": 42, "verbosity": 0},
            "search_space": {"n_estimators": [10, 20], "max_depth": [2, 3]},
        },
        "catboost": {
            "fixed_params": {"random_seed": 42, "verbose": False},
            "search_space": {"iterations": [10, 20], "depth": [3, 4]},
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{db_path}",
            "experiment_name": "test_dynamic_pricing",
            "registered_model_name": "test_dynamic_pricing_model",
            "production_alias": "production",
        },
    }



@pytest.fixture
def feature_table():
    rng = np.random.default_rng(7)
    n = 150
    duration = rng.uniform(10, 180, n)
    extra_feature = rng.uniform(0, 1, n)
    cost = duration * 4.5 + rng.normal(0, 5, n)
    return pd.DataFrame({"duration": duration, "extra_feature": extra_feature, "cost": cost})


def test_full_training_chain_runs_end_to_end_with_real_mlflow(feature_table, parameters):
    X_train, X_test, y_train, y_test = split_data(feature_table, parameters)

    baseline_model = train_baseline_model(X_train, y_train, parameters)
    random_forest_model = train_random_forest(X_train, y_train, parameters)
    lightgbm_model = train_lightgbm_model(X_train, y_train, parameters)
    xgboost_model = train_xgboost_model(X_train, y_train, parameters)
    catboost_model = train_catboost_model(X_train, y_train, parameters)

    models = collect_trained_models(
        baseline_model, random_forest_model, lightgbm_model, xgboost_model, catboost_model
    )
    assert len(models) == 5

    comparison_table = evaluate_models(models, X_test, y_test, parameters)
    assert len(comparison_table) == 5

    summary = log_and_register_best_model(models, comparison_table, X_train, parameters)

    assert summary["best_model_name"] in models
    assert summary["best_model_version"] is not None
    assert "rmse" in summary["metrics"]

    # Verify the registration and alias actually landed in the MLflow registry,
    # not just that the node returned without raising.
    client = MlflowClient(tracking_uri=parameters["mlflow"]["tracking_uri"])
    registered_model_name = parameters["mlflow"]["registered_model_name"]

    all_versions = client.search_model_versions(f"name='{registered_model_name}'")
    assert len(all_versions) == 5, "every one of the 5 models should be a registered version"

    production_version = client.get_model_version_by_alias(registered_model_name, "production")
    assert production_version.version == summary["best_model_version"]

    print(f"Best model: {summary['best_model_name']} (version {summary['best_model_version']})")
    print(comparison_table)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
