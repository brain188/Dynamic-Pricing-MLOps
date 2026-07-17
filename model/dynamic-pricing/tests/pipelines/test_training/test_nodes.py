"""Unit tests for ``dynamic_pricing.pipelines.training.nodes``.

Model training/evaluation nodes are tested against small synthetic
regression data rather than the real dataset, so tests run fast and
failures point precisely at broken logic. MLflow-touching logic
(`log_and_register_best_model`) is tested separately in
`test_integration.py` against a real (temporary) MLflow tracking store,
since mocking MLflow's client would not actually verify the registration
workflow works.
"""

import numpy as np
import pandas as pd
import pytest

from dynamic_pricing.pipelines.training.nodes import (
    _compute_regression_metrics,
    collect_trained_models,
    evaluate_models,
    split_data,
    train_baseline_model,
    train_catboost_model,
    train_lightgbm_model,
    train_random_forest,
    train_xgboost_model,
)


@pytest.fixture
def base_parameters():
    """Small search spaces and n_iter/cv_folds keep unit tests fast — the
    goal here is verifying the tuning plumbing works, not finding a truly
    optimal model on toy data."""
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
    }



@pytest.fixture
def feature_table():
    rng = np.random.default_rng(42)
    n = 200
    duration = rng.uniform(10, 180, n)
    extra_feature = rng.uniform(0, 1, n)
    cost = duration * 4.5 + rng.normal(0, 5, n)  # duration dominates, matching real EDA finding
    return pd.DataFrame({"duration": duration, "extra_feature": extra_feature, "cost": cost})


class TestSplitData:
    def test_split_sizes_match_test_size(self, feature_table, base_parameters):
        X_train, X_test, y_train, y_test = split_data(feature_table, base_parameters)
        assert len(X_test) == round(len(feature_table) * 0.25)
        assert len(X_train) + len(X_test) == len(feature_table)

    def test_target_column_not_in_features(self, feature_table, base_parameters):
        X_train, X_test, _, _ = split_data(feature_table, base_parameters)
        assert "cost" not in X_train.columns
        assert "cost" not in X_test.columns


class TestTrainBaselineModel:
    def test_fits_only_on_baseline_columns(self, feature_table, base_parameters):
        X_train, _, y_train, _ = split_data(feature_table, base_parameters)
        model = train_baseline_model(X_train, y_train, base_parameters)
        assert model.n_features_in_ == 1

    def test_learns_a_reasonable_relationship(self, feature_table, base_parameters):
        X_train, _, y_train, _ = split_data(feature_table, base_parameters)
        model = train_baseline_model(X_train, y_train, base_parameters)
        # cost ~= duration * 4.5, so the learned coefficient should be close to that
        assert model.coef_[0] == pytest.approx(4.5, abs=0.5)


class TestTrainCandidateModels:
    def test_random_forest_trains_and_predicts(self, feature_table, base_parameters):
        X_train, X_test, y_train, _ = split_data(feature_table, base_parameters)
        model = train_random_forest(X_train, y_train, base_parameters)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_lightgbm_trains_and_predicts(self, feature_table, base_parameters):
        X_train, X_test, y_train, _ = split_data(feature_table, base_parameters)
        model = train_lightgbm_model(X_train, y_train, base_parameters)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_xgboost_trains_and_predicts(self, feature_table, base_parameters):
        X_train, X_test, y_train, _ = split_data(feature_table, base_parameters)
        model = train_xgboost_model(X_train, y_train, base_parameters)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_catboost_trains_and_predicts(self, feature_table, base_parameters):
        X_train, X_test, y_train, _ = split_data(feature_table, base_parameters)
        model = train_catboost_model(X_train, y_train, base_parameters)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestComputeRegressionMetrics:
    def test_perfect_predictions_give_zero_error(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        metrics = _compute_regression_metrics(y_true, y_true)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)


class TestEvaluateModels:
    def test_returns_one_row_per_model_sorted_by_rmse(self, feature_table, base_parameters):
        X_train, X_test, y_train, y_test = split_data(feature_table, base_parameters)
        baseline = train_baseline_model(X_train, y_train, base_parameters)
        rf = train_random_forest(X_train, y_train, base_parameters)
        models = collect_trained_models(baseline, rf, rf, rf, rf)  # reuse rf as stand-ins for speed

        comparison = evaluate_models(models, X_test, y_test, base_parameters)

        assert len(comparison) == len(models)
        assert list(comparison.columns) == ["model_name", "rmse", "mae", "r2"]
        assert comparison["rmse"].is_monotonic_increasing

    def test_baseline_evaluated_only_on_baseline_columns(self, feature_table, base_parameters):
        X_train, X_test, y_train, y_test = split_data(feature_table, base_parameters)
        baseline = train_baseline_model(X_train, y_train, base_parameters)
        models = {"linear_regression_baseline": baseline}
        # Should not raise, even though X_test has an extra column baseline wasn't fit on.
        comparison = evaluate_models(models, X_test, y_test, base_parameters)
        assert len(comparison) == 1


class TestCollectTrainedModels:
    def test_returns_correctly_keyed_dict(self):
        models = collect_trained_models("a", "b", "c", "d", "e")
        assert models == {
            "linear_regression_baseline": "a",
            "random_forest": "b",
            "lightgbm": "c",
            "xgboost": "d",
            "catboost": "e",
        }
