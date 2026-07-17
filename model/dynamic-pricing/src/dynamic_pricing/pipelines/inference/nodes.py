"""Node functions for the ``inference`` pipeline.

This module scores new, unseen ride requests using the model currently
aliased ``"production"`` in the MLflow Model Registry. Only three things
happen here, and only here:

    1. Loading the production model by registry alias (never a hardcoded
       version number, so promoting a new winner requires no code change).
    2. Generating predictions, with the model's own MLflow signature
       determining which columns it actually needs — this pipeline never
       hardcodes "the production model uses all features" or "just
       duration," because that has already changed once (the baseline
       won initially, then CatBoost won after retuning) and will likely
       change again after future retraining.
    3. Sanity-checking predictions against business-sane bounds

Everything *before* scoring — schema validation, demand-supply ratio
engineering, categorical encoding — is deliberately **not** reimplemented
here. This pipeline's ``pipeline.py`` wires in the exact same node
functions from ``data_processing`` and ``feature_engineering`` directly,
reusing the same code and the same persisted, fitted
``categorical_encoder`` artifact that training used. This is what
guarantees training and serving compute features identically (the
"consistent features between training and production" MLOps concern) —
it's enforced by reusing the literal functions, not by two
independently-written implementations that could quietly drift apart.
"""

import logging
from typing import Any

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def load_production_model(parameters: dict[str, Any]) -> mlflow.pyfunc.PyFuncModel:

    """Loads the model currently aliased ``"production"`` from the MLflow
    Model Registry.

    Args:
        parameters: The ``inference`` parameters dictionary. Must contain
            an ``mlflow`` block with ``tracking_uri``,
            ``registered_model_name``, and ``production_alias``.

    Returns:
        The loaded model as an MLflow ``pyfunc`` object, ready for
        ``.predict()``. Loading it as a generic ``pyfunc`` (rather than,
        say, ``mlflow.catboost.load_model``) means this pipeline works
        unchanged regardless of which underlying library produced the
        currently-promoted model.
    """
    mlflow_config = parameters["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    model_uri = (
        f"models:/{mlflow_config['registered_model_name']}"
        f"@{mlflow_config['production_alias']}"
    )
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info(
        "Loaded production model from '%s' (run_id=%s, flavor=%s).",
        model_uri,
        model.metadata.run_id,
        list(model.metadata.flavors.keys()),
    )

    return model


def generate_predictions(
    encoded_ride_requests: pd.DataFrame,
    production_model: mlflow.pyfunc.PyFuncModel,
    parameters: dict[str, Any],
) -> pd.DataFrame:

    """Generates fare predictions using the production model.

    The model's own MLflow signature (logged automatically from the
    ``input_example`` passed during training) determines which columns
    are actually fed to ``.predict()``. This is what lets this function
    work correctly whether the production model is the duration-only
    baseline or a full-feature model — the column selection is read from
    the model itself, never hardcoded.

    Args:
        encoded_ride_requests: Fully feature-engineered ride request data
            (demand-supply ratio computed, categoricals one-hot encoded) —
            the output of reusing ``feature_engineering``'s node functions
            within this pipeline.
        production_model: The loaded MLflow ``pyfunc`` model, as returned
            by ``load_production_model``.
        parameters: The ``inference`` parameters dictionary. Must contain
            ``prediction_column_name``.

    Returns:
        The input dataframe with a new column (named per
        ``prediction_column_name``) containing the predicted fare for
        each row.

    Raises:
        ValueError: If the model's signature declares required input
            columns that are missing from ``encoded_ride_requests`` —
            this indicates a mismatch between how features were
            engineered here and how the model was trained, and must not
            be silently ignored.
    """
    prediction_column = parameters["prediction_column_name"]
    input_schema = production_model.metadata.get_input_schema()

    if input_schema is not None:
        required_columns = input_schema.input_names()
        missing_columns = set(required_columns) - set(encoded_ride_requests.columns)
        if missing_columns:
            logger.error(
                "Model expects columns %s but they are missing from the "
                "engineered inference data. Available columns: %s.",
                sorted(missing_columns),
                sorted(encoded_ride_requests.columns),
            )
            raise ValueError(
                f"Production model requires column(s) {sorted(missing_columns)} "
                "that are not present after feature engineering. This indicates "
                "training/serving feature skew and must be investigated before "
                "predictions can be trusted."
            )
        model_input = encoded_ride_requests[required_columns]
    else:
        logger.warning(
            "Production model has no logged input signature; passing all "
            "engineered columns to predict(). Consider always logging an "
            "input_example during training to avoid this fallback."
        )
        model_input = encoded_ride_requests

    predictions = production_model.predict(model_input)

    result = encoded_ride_requests.copy()
    result[prediction_column] = predictions

    logger.info(
        "Generated %d prediction(s) using %d input column(s). "
        "Prediction range: [%.2f, %.2f], mean: %.2f.",
        len(result),
        model_input.shape[1],
        result[prediction_column].min(),
        result[prediction_column].max(),
        result[prediction_column].mean(),
    )

    return result


def validate_predictions(
    predictions: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Sanity-checks predictions against business-sane bounds.

    A predicted fare that is negative, or absurdly large relative to any
    fare seen historically, indicates a problem upstream (bad input data,
    a feature engineering bug, or a genuinely broken model) — this node
    exists to catch that before predictions reach a rider or an operator.

    Args:
        predictions: The dataframe returned by ``generate_predictions``.
        parameters: The ``inference`` parameters dictionary. Must contain
            ``prediction_column_name`` and a ``sanity_bounds`` block with
            ``min``, ``max``, and ``action`` (one of ``"clip"`` or
            ``"flag"``).

    Returns:
        The predictions dataframe. If ``action`` is ``"clip"``,
        out-of-bounds predictions are clipped to the nearest bound. If
        ``"flag"``, out-of-bounds predictions are left as-is but logged
        as warnings, and a boolean ``is_out_of_bounds`` column is added
        so downstream consumers (e.g. the monitoring pipeline) can filter
        on it.
    """
    prediction_column = parameters["prediction_column_name"]
    bounds = parameters["sanity_bounds"]
    lower, upper, action = bounds["min"], bounds["max"], bounds["action"]

    out_of_bounds_mask = (predictions[prediction_column] < lower) | (
        predictions[prediction_column] > upper
    )
    n_out_of_bounds = int(out_of_bounds_mask.sum())

    result = predictions.copy()

    if n_out_of_bounds == 0:
        logger.info(
            "All %d prediction(s) within sane bounds [%.2f, %.2f].",
            len(result),
            lower,
            upper,
        )
        result["is_out_of_bounds"] = False
        return result

    logger.warning(
        "%d prediction(s) (%.2f%% of batch) fell outside sane bounds [%.2f, %.2f]. "
        "action='%s'.",
        n_out_of_bounds,
        n_out_of_bounds / len(result) * 100,
        lower,
        upper,
        action,
    )

    if action == "clip":
        result[prediction_column] = result[prediction_column].clip(lower=lower, upper=upper)
        result["is_out_of_bounds"] = False
        logger.info("Out-of-bounds predictions clipped to [%.2f, %.2f].", lower, upper)
    elif action == "flag":
        result["is_out_of_bounds"] = out_of_bounds_mask
    else:
        raise ValueError(f"Unsupported sanity_bounds.action '{action}'. Expected 'clip' or 'flag'.")

    return result
