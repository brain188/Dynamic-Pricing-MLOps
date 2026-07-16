"""Node functions for the ``feature_engineering`` pipeline.

This module turns the clean dataset produced by ``data_processing`` into a
model-ready feature table. Two things happen here, and only here:

    * Engineering the ``demand_supply_ratio`` feature (riders ÷ drivers),
      including outlier capping — the central hypothesis of this whole
      project, validated during EDA (see ``modeling.ipynb`` and the
      permutation importance follow-up).
    * Encoding categorical features into a numeric, model-ready form.

"""

import logging
from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def engineer_demand_supply_ratio(
    cleaned_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Engineers the demand-supply ratio feature and caps its outliers.

    The ratio is computed as ``numerator_column / denominator_column``
    (riders ÷ drivers by default). EDA found this ratio has ~7% outliers
    by the IQR method — driven by rows with very low driver counts — so
    values above a configured percentile are capped (winsorized) rather
    than removed, to avoid losing otherwise-valid rows over one feature.

    Args:
        cleaned_data: The validated, deduplicated dataset produced by the
            ``data_processing`` pipeline.
        parameters: The ``feature_engineering`` parameters dictionary.
            Must contain a ``demand_supply_ratio`` block with
            ``numerator_column``, ``denominator_column``, and
            ``outlier_cap_percentile`` (a float in (0, 1]).

    Returns:
        The input dataframe with a new ``demand_supply_ratio`` column
        appended. All original columns are preserved.

    Raises:
        ValueError: If the denominator column contains zero values, which
            would produce an infinite ratio.
    """
    config = parameters["demand_supply_ratio"]
    numerator_col = config["numerator_column"]
    denominator_col = config["denominator_column"]
    cap_percentile = config["outlier_cap_percentile"]

    if (cleaned_data[denominator_col] == 0).any():
        n_zero = int((cleaned_data[denominator_col] == 0).sum())
        logger.error(
            "%d row(s) have %s == 0, which would produce an infinite ratio.",
            n_zero,
            denominator_col,
        )
        raise ValueError(
            f"Column '{denominator_col}' contains {n_zero} zero value(s); "
            "cannot compute demand_supply_ratio. Handle zero-driver rows "
            "upstream in data_processing before reaching feature_engineering."
        )

    df = cleaned_data.copy()
    raw_ratio = df[numerator_col] / df[denominator_col]

    cap_value = raw_ratio.quantile(cap_percentile)
    n_capped = int((raw_ratio > cap_value).sum())

    df["demand_supply_ratio"] = raw_ratio.clip(upper=cap_value)

    logger.info(
        "Engineered demand_supply_ratio (%s / %s). Capped %d row(s) "
        "(%.2f%% of dataset) above the %.0fth percentile value of %.3f.",
        numerator_col,
        denominator_col,
        n_capped,
        (n_capped / len(df) * 100) if len(df) else 0.0,
        cap_percentile * 100,
        cap_value,
    )

    return df


def fit_categorical_encoder(
    data_with_ratio: pd.DataFrame, parameters: dict[str, Any]
) -> OneHotEncoder:

    """Fits a one-hot encoder on the dataset's categorical columns.

    The fitted encoder is returned as its own artifact so it can be
    persisted (via the ``categorical_encoder`` catalog entry) and reused
    unchanged by the inference pipeline later — this is what guarantees
    training and serving compute features the same way.

    Args:
        data_with_ratio: The dataset after demand-supply ratio engineering.
        parameters: The ``feature_engineering`` parameters dictionary.
            Must contain ``categorical_columns`` (list of column names) and
            an ``encoder`` block with ``handle_unknown`` and ``drop``
            options matching ``sklearn.preprocessing.OneHotEncoder``'s
            constructor arguments.

    Returns:
        A fitted ``OneHotEncoder`` instance.
    """
    categorical_columns: list[str] = parameters["categorical_columns"]
    encoder_config = parameters["encoder"]

    encoder = OneHotEncoder(
        handle_unknown=encoder_config["handle_unknown"],
        drop=encoder_config["drop"],
        sparse_output=False,
    )
    encoder.fit(data_with_ratio[categorical_columns])

    n_output_columns = len(encoder.get_feature_names_out(categorical_columns))
    logger.info(
        "Fitted OneHotEncoder on %d categorical column(s): %s. "
        "Produces %d encoded column(s). handle_unknown='%s', drop=%s.",
        len(categorical_columns),
        categorical_columns,
        n_output_columns,
        encoder_config["handle_unknown"],
        encoder_config["drop"],
    )

    return encoder


def encode_categorical_features(
    data_with_ratio: pd.DataFrame,
    categorical_encoder: OneHotEncoder,
    parameters: dict[str, Any],
) -> pd.DataFrame:

    """Applies a fitted one-hot encoder to the dataset's categorical columns.

    Original categorical columns are dropped and replaced with their
    one-hot encoded equivalents. Non-categorical columns (numeric features,
    the engineered ratio, and the target) pass through unchanged.

    Args:
        data_with_ratio: The dataset after demand-supply ratio engineering.
        categorical_encoder: A fitted ``OneHotEncoder``, as produced by
            ``fit_categorical_encoder``. Passing a fitted encoder in (rather
            than fitting inside this function) is what allows this same
            function to be reused, unchanged, by the inference pipeline
            with a loaded (not re-fit) encoder.
        parameters: The ``feature_engineering`` parameters dictionary. Must
            contain ``categorical_columns`` (list of column names).

    Returns:
        The dataframe with categorical columns replaced by their one-hot
        encoded columns.
    """
    categorical_columns: list[str] = parameters["categorical_columns"]

    encoded_array = categorical_encoder.transform(data_with_ratio[categorical_columns])
    encoded_column_names = categorical_encoder.get_feature_names_out(categorical_columns)

    encoded_df = pd.DataFrame(
        encoded_array, columns=encoded_column_names, index=data_with_ratio.index
    )

    passthrough_df = data_with_ratio.drop(columns=categorical_columns)
    result = pd.concat([passthrough_df, encoded_df], axis=1)

    logger.info(
        "Encoded %d categorical column(s) into %d one-hot column(s). "
        "Output shape: %s (was %s before encoding).",
        len(categorical_columns),
        len(encoded_column_names),
        result.shape,
        data_with_ratio.shape,
    )

    return result


def assemble_feature_table(
    encoded_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Assembles the final, model-ready feature table.

    Performs final housekeeping: places the target column last (a
    readability convention, not a functional requirement), verifies no
    nulls were introduced during encoding, and logs the final schema so
    the feature table's shape is always visible in pipeline run logs.

    Args:
        encoded_data: The dataset after categorical encoding.
        parameters: The ``feature_engineering`` parameters dictionary. Must
            contain ``target_column`` (the name of the prediction target).

    Returns:
        The final feature table, ready to be consumed by the ``training``
        pipeline (with the target column present) and, after dropping the
        target, by the ``inference`` pipeline.

    Raises:
        ValueError: If any null values are present in the assembled table,
            since nulls at this stage indicate a bug earlier in the
            pipeline rather than an expected data condition.
    """
    target_column: str = parameters["target_column"]

    if encoded_data.isnull().any().any():
        null_summary = encoded_data.isnull().sum()
        null_columns = null_summary[null_summary > 0].to_dict()
        logger.error("Unexpected nulls found in assembled feature table: %s", null_columns)
        raise ValueError(
            f"Feature table contains unexpected null values: {null_columns}. "
            "This indicates a bug upstream (data_processing or encoding), "
            "not an expected data condition at this stage."
        )

    feature_columns = [col for col in encoded_data.columns if col != target_column]
    feature_table = encoded_data[feature_columns + [target_column]]

    logger.info(
        "Assembled final feature table: %d rows, %d feature column(s) + 1 target column.\n"
        "Feature columns: %s",
        len(feature_table),
        len(feature_columns),
        feature_columns,
    )

    return feature_table
