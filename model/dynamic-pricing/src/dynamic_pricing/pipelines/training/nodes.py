"""Node functions for the ``training`` pipeline.

This module trains and compares five regression models against the
feature table produced by ``feature_engineering``:

    1. Linear Regression, duration-only  (the mandatory baseline)
    2. Random Forest, all features       (regularized via cross-validated search)
    3. LightGBM, all features            (regularized via cross-validated search)
    4. XGBoost, all features             (regularized via cross-validated search)
    5. CatBoost, all features            (regularized via cross-validated search)

The four non-baseline models are tuned with ``RandomizedSearchCV`` over a
*regularization-leaning* search space (shallower trees, fewer leaves,
stronger L1/L2 penalties, subsampling) rather than fixed hand-picked
hyperparameters. This exists because an initial fixed-hyperparameter run
found every tree/boosting model underperforming the duration-only
baseline — a signature of overfitting on this project's small (1000-row)
dataset and its several low-signal categorical features (confirmed weak
by both impurity-based and permutation feature importance during EDA).
Searching toward more conservative configurations is the correct next
step to check whether that gap is genuine (the true relationship really
is closely linear) or just undertuning.

Every model is logged to MLflow (Experiment Tracking)
as its own run, with parameters, test-set metrics, and the fitted model
artifact. The model with the best test-set RMSE is then registered in the
MLflow Model Registry and aliased ``"production"`` — this alias is exactly
what the ``inference`` pipeline and the FastAPI backend will resolve at
load time, so promoting a new winner later is a registry operation, not a code change.

"""

import logging
from typing import Any

import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def split_data(
    feature_table: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    """Splits the feature table into train and test sets.

    Args:
        feature_table: The model-ready feature table from
            ``feature_engineering``.
        parameters: The ``training`` parameters dictionary. Must contain
            ``target_column``, ``test_size``, and ``random_state``.

    Returns:
        A tuple of ``(X_train, X_test, y_train, y_test)``.
    """
    target_column = parameters["target_column"]
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    X = feature_table.drop(columns=[target_column])
    y = feature_table[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Split feature table (%d rows) into train (%d rows) and test (%d rows), "
        "test_size=%.2f, random_state=%d.",
        len(feature_table),
        len(X_train),
        len(X_test),
        test_size,
        random_state,
    )

    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> LinearRegression:

    """Trains the mandatory duration-only Linear Regression baseline.

    This model mirrors the company's *current* pricing approach (fare
    determined by ride duration alone) and is the benchmark every other
    model must be compared against.

    Args:
        X_train: Training features (only the configured baseline column(s)
            are used).
        y_train: Training target values.
        parameters: The ``training`` parameters dictionary. Must contain
            ``baseline_feature_columns`` (a list of column names).

    Returns:
        A fitted ``LinearRegression`` model.
    """
    y_train = y_train.squeeze()
    baseline_columns = parameters["baseline_feature_columns"]

    model = LinearRegression()
    model.fit(X_train[baseline_columns], y_train)

    logger.info(
        "Trained baseline Linear Regression on %s. Coefficients: %s, intercept: %.4f.",
        baseline_columns,
        dict(zip(baseline_columns, model.coef_.round(4))),
        model.intercept_,
    )

    return model


def _tune_regressor(
    estimator_class: type[BaseEstimator],
    fixed_params: dict[str, Any],
    search_space: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_config: dict[str, Any],
    model_label: str,
) -> BaseEstimator:

    """Runs a cross-validated randomized hyperparameter search and refits
    the best configuration on the full training set.

    Shared by all four tree/boosting models so each one is tuned with the
    exact same search procedure (only the estimator class, fixed params,
    and search space differ), rather than four subtly different tuning
    implementations.

    Args:
        estimator_class: The unfitted regressor class to tune (e.g.
            ``RandomForestRegressor``).
        fixed_params: Keyword arguments held constant across the search
            (e.g. ``random_state``, ``n_jobs``) — not tuned.
        search_space: A mapping of hyperparameter name to a list of
            candidate values, passed to ``RandomizedSearchCV`` as
            ``param_distributions``. Intentionally regularization-leaning
            (shallower trees, stronger penalties) per this module's
            docstring.
        X_train: Training features.
        y_train: Training target values.
        tuning_config: The shared ``training.tuning`` parameters block.
            Must contain ``cv_folds``, ``n_iter``, ``scoring``, and
            ``random_state``.
        model_label: A human-readable name used only for logging.

    Returns:
        The best estimator found, refit on the entire training set
        (``RandomizedSearchCV(refit=True)`` does this automatically).
    """
    base_estimator = estimator_class(**fixed_params)

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=search_space,
        n_iter=tuning_config["n_iter"],
        cv=tuning_config["cv_folds"],
        scoring=tuning_config["scoring"],
        random_state=tuning_config["random_state"],
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Tuned '%s' via %d-fold RandomizedSearchCV (%d candidates tried). "
        "Best CV %s: %.4f. Best params: %s.",
        model_label,
        tuning_config["cv_folds"],
        tuning_config["n_iter"],
        tuning_config["scoring"],
        search.best_score_,
        search.best_params_,
    )

    return search.best_estimator_


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> RandomForestRegressor:

    """Tunes and trains a Random Forest regressor on all features.

    Args:
        X_train: Training features (full feature set).
        y_train: Training target values.
        parameters: The ``training`` parameters dictionary. Must contain a
            ``random_forest`` block with ``fixed_params`` and
            ``search_space`` sub-blocks, and a shared ``tuning`` block.

    Returns:
        The best-tuned, fully-refit ``RandomForestRegressor``.
    """
    y_train = y_train.squeeze()
    config = parameters["random_forest"]

    return _tune_regressor(
        estimator_class=RandomForestRegressor,
        fixed_params=config["fixed_params"],
        search_space=config["search_space"],
        X_train=X_train,
        y_train=y_train,
        tuning_config=parameters["tuning"],
        model_label="random_forest",
    )


def train_lightgbm_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> LGBMRegressor:

    """Tunes and trains a LightGBM regressor on all features.

    Args:
        X_train: Training features (full feature set).
        y_train: Training target values.
        parameters: The ``training`` parameters dictionary. Must contain a
            ``lightgbm`` block with ``fixed_params`` and ``search_space``
            sub-blocks, and a shared ``tuning`` block.

    Returns:
        The best-tuned, fully-refit ``LGBMRegressor``.
    """
    y_train = y_train.squeeze()
    config = parameters["lightgbm"]

    return _tune_regressor(
        estimator_class=LGBMRegressor,
        fixed_params=config["fixed_params"],
        search_space=config["search_space"],
        X_train=X_train,
        y_train=y_train,
        tuning_config=parameters["tuning"],
        model_label="lightgbm",
    )


def train_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> XGBRegressor:

    """Tunes and trains an XGBoost regressor on all features.

    Args:
        X_train: Training features (full feature set).
        y_train: Training target values.
        parameters: The ``training`` parameters dictionary. Must contain an
            ``xgboost`` block with ``fixed_params`` and ``search_space``
            sub-blocks, and a shared ``tuning`` block.

    Returns:
        The best-tuned, fully-refit ``XGBRegressor``.
    """
    y_train = y_train.squeeze()
    config = parameters["xgboost"]

    return _tune_regressor(
        estimator_class=XGBRegressor,
        fixed_params=config["fixed_params"],
        search_space=config["search_space"],
        X_train=X_train,
        y_train=y_train,
        tuning_config=parameters["tuning"],
        model_label="xgboost",
    )


def train_catboost_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> CatBoostRegressor:

    """Tunes and trains a CatBoost regressor on all features.

    Args:
        X_train: Training features (full feature set).
        y_train: Training target values.
        parameters: The ``training`` parameters dictionary. Must contain a
            ``catboost`` block with ``fixed_params`` and ``search_space``
            sub-blocks, and a shared ``tuning`` block.

    Returns:
        The best-tuned, fully-refit ``CatBoostRegressor``.
    """
    y_train = y_train.squeeze()
    config = parameters["catboost"]

    return _tune_regressor(
        estimator_class=CatBoostRegressor,
        fixed_params=config["fixed_params"],
        search_space=config["search_space"],
        X_train=X_train,
        y_train=y_train,
        tuning_config=parameters["tuning"],
        model_label="catboost",
    )


def collect_trained_models(
    baseline_model: LinearRegression,
    random_forest_model: RandomForestRegressor,
    lightgbm_model: LGBMRegressor,
    xgboost_model: XGBRegressor,
    catboost_model: CatBoostRegressor,
) -> dict[str, Any]:

    """Collects all five trained models into a single named dictionary.

    Args:
        baseline_model: The fitted Linear Regression baseline.
        random_forest_model: The fitted Random Forest model.
        lightgbm_model: The fitted LightGBM model.
        xgboost_model: The fitted XGBoost model.
        catboost_model: The fitted CatBoost model.

    Returns:
        A dictionary mapping model name to fitted model object. Keys match
        the ``model_names`` list expected elsewhere in this pipeline's
        parameters (``linear_regression_baseline``, ``random_forest``,
        ``lightgbm``, ``xgboost``, ``catboost``).
    """
    return {
        "linear_regression_baseline": baseline_model,
        "random_forest": random_forest_model,
        "lightgbm": lightgbm_model,
        "xgboost": xgboost_model,
        "catboost": catboost_model,
    }


def _compute_regression_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:

    """Computes RMSE, MAE, and R² for a set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.

    Returns:
        A dictionary with keys ``rmse``, ``mae``, and ``r2``.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    parameters: dict[str, Any],
) -> pd.DataFrame:

    """Evaluates every trained model on the held-out test set.

    The baseline model is evaluated only on its configured subset of
    columns (matching how it was trained); all other models are evaluated
    on the full feature set.

    Args:
        models: A dictionary of {model_name: fitted_model}, as produced by
            ``collect_trained_models``.
        X_test: Held-out test features.
        y_test: Held-out test target values.
        parameters: The ``training`` parameters dictionary. Must contain
            ``baseline_model_key`` and ``baseline_feature_columns``.

    Returns:
        A dataframe with one row per model, columns
        ``["model_name", "rmse", "mae", "r2"]``, sorted by RMSE ascending
        (best model first).
    """
    y_test = y_test.squeeze()
    baseline_key = parameters["baseline_model_key"]
    baseline_columns = parameters["baseline_feature_columns"]

    rows = []
    for model_name, model in models.items():
        X_eval = X_test[baseline_columns] if model_name == baseline_key else X_test
        predictions = model.predict(X_eval)
        metrics = _compute_regression_metrics(y_test, predictions)
        rows.append({"model_name": model_name, **metrics})

        logger.info(
            "Evaluated '%s': RMSE=%.3f, MAE=%.3f, R2=%.4f.",
            model_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )

    comparison_table = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)

    baseline_rmse = comparison_table.loc[
        comparison_table["model_name"] == baseline_key, "rmse"
    ].iloc[0]
    best_row = comparison_table.iloc[0]
    improvement_pct = (baseline_rmse - best_row["rmse"]) / baseline_rmse * 100

    logger.info(
        "Best model: '%s' (RMSE=%.3f) vs. baseline RMSE=%.3f -> %.2f%% improvement.",
        best_row["model_name"],
        best_row["rmse"],
        baseline_rmse,
        improvement_pct,
    )

    return comparison_table


_MLFLOW_LOG_MODEL_FN = {
    "linear_regression_baseline": mlflow.sklearn.log_model,
    "random_forest": mlflow.sklearn.log_model,
    "lightgbm": mlflow.lightgbm.log_model,
    "xgboost": mlflow.xgboost.log_model,
    "catboost": mlflow.catboost.log_model,
}


def log_and_register_best_model(
    models: dict[str, Any],
    comparison_table: pd.DataFrame,
    X_train: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Logs every model's run to MLflow and promotes the best one.

    Each model is logged as its own MLflow run (parameters + test metrics
    + model artifact, using the correct flavor per model type). Every run
    also registers a new version of the same registered model name in the
    MLflow Model Registry — this is expected and intentional: the registry
    is meant to hold every candidate version, with the ``"production"``
    alias marking which one is actually promoted for serving.

    Args:
        models: A dictionary of {model_name: fitted_model}.
        comparison_table: The test-set metrics table from
            ``evaluate_models``, used both for per-run metric logging and
            to determine the winner.
        X_train: Training features, used to build an MLflow model
            signature/input example for each logged model.
        parameters: The ``training`` parameters dictionary. Must contain an
            ``mlflow`` block with ``tracking_uri``, ``experiment_name``,
            ``registered_model_name``, and ``production_alias``, plus
            ``baseline_model_key`` and ``baseline_feature_columns``.

    Returns:
        A summary dictionary with keys ``best_model_name``,
        ``best_model_version``, and ``metrics`` (the winning model's
        RMSE/MAE/R2), for logging and for the training pipeline's final
        output artifact.
    """
    mlflow_config = parameters["mlflow"]
    baseline_key = parameters["baseline_model_key"]
    baseline_columns = parameters["baseline_feature_columns"]
    registered_model_name = mlflow_config["registered_model_name"]

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    client = MlflowClient()
    run_id_by_model_name = {}

    for _, row in comparison_table.iterrows():
        model_name = row["model_name"]
        model = models[model_name]
        log_model_fn = _MLFLOW_LOG_MODEL_FN[model_name]
        input_example = X_train[baseline_columns].head(5) if model_name == baseline_key else X_train.head(5)

        with mlflow.start_run(run_name=model_name, nested=True) as run:
            mlflow.log_metrics(
                {"rmse": row["rmse"], "mae": row["mae"], "r2": row["r2"]}
            )
            if hasattr(model, "get_params"):
                # Log only JSON/primitive-serializable params; skip anything
                # mlflow can't handle (e.g. nested objects) rather than crash.
                safe_params = {
                    k: v for k, v in model.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))
                }
                mlflow.log_params(safe_params)

            log_model_fn(
                model,
                name="model",
                registered_model_name=registered_model_name,
                input_example=input_example,
            )
            run_id_by_model_name[model_name] = run.info.run_id

        logger.info(
            "Logged '%s' to MLflow experiment '%s' (run_id=%s).",
            model_name,
            mlflow_config["experiment_name"],
            run.info.run_id,
        )

    best_model_name = comparison_table.iloc[0]["model_name"]
    best_run_id = run_id_by_model_name[best_model_name]

    versions = client.search_model_versions(f"name='{registered_model_name}'")
    best_version = next(v for v in versions if v.run_id == best_run_id)

    client.set_registered_model_alias(
        registered_model_name, mlflow_config["production_alias"], best_version.version
    )

    logger.info(
        "Registered '%s' (version %s) as '%s' alias for model '%s'.",
        best_model_name,
        best_version.version,
        mlflow_config["production_alias"],
        registered_model_name,
    )

    best_metrics = comparison_table.iloc[0][["rmse", "mae", "r2"]].to_dict()

    return {
        "best_model_name": best_model_name,
        "best_model_version": best_version.version,
        "metrics": best_metrics,
    }
