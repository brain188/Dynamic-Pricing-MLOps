"""Node functions for the ``data_processing`` pipeline.

This module contains the logic that turns raw ride-sharing data into a
validated, clean dataset ready for feature engineering. It is intentionally
scoped to *data quality* concerns only:

    * schema validation (expected columns and dtypes are present)
    * duplicate removal
    * missing value handling
    * value-range sanity checks (e.g. ratings between 1 and 5)
    * dtype enforcement

No feature engineering (e.g. the demand-supply ratio) happens here — that
belongs to the ``feature_engineering`` pipeline, which consumes the output
of this one. Keeping this separation means the data_processing pipeline can
be reused even if the feature engineering strategy changes later.

"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(
    raw_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Validates that the raw dataset has the expected columns and dtypes.

    This is a fail-fast guardrail: if an upstream data source changes its
    schema (a column is renamed, dropped, or its type changes), this node
    raises immediately rather than letting a malformed dataset silently
    flow into feature engineering and training.

    Args:
        raw_data: The raw ride-sharing dataset as loaded from the catalog.
        parameters: The ``data_processing`` parameters dictionary. Must
            contain an ``expected_schema`` mapping of column name to
            expected pandas dtype string (e.g. ``"int64"``, ``"float64"``,
            ``"object"``).

    Returns:
        The same dataframe, unmodified, if validation passes.

    Raises:
        ValueError: If any expected column is missing, if unexpected extra
            columns are present, or if a column's dtype does not match
            the expected dtype declared in parameters.
    """
    expected_schema: dict[str, str] = parameters["expected_schema"]
    expected_columns = set(expected_schema.keys())
    actual_columns = set(raw_data.columns)

    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - expected_columns

    if missing_columns:
        logger.error("Missing expected columns: %s", sorted(missing_columns))
        raise ValueError(
            f"Raw data is missing expected columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        logger.warning(
            "Raw data contains unexpected extra columns: %s. "
            "These will be carried through unless explicitly dropped.",
            sorted(unexpected_columns),
        )

    string_like_dtypes = {"object", "str"}
    dtype_mismatches = {}
    for column, expected_dtype in expected_schema.items():
        actual_dtype = str(raw_data[column].dtype)
        is_equivalent_string_dtype = (
            expected_dtype in string_like_dtypes and actual_dtype in string_like_dtypes
        )
        if actual_dtype != expected_dtype and not is_equivalent_string_dtype:
            dtype_mismatches[column] = (expected_dtype, actual_dtype)

    if dtype_mismatches:
        logger.error("Dtype mismatches found: %s", dtype_mismatches)
        raise ValueError(
            f"Column dtype mismatches (expected, actual): {dtype_mismatches}"
        )

    logger.info(
        "Schema validation passed: %d columns, %d rows.",
        len(actual_columns),
        len(raw_data),
    )
    return raw_data


def remove_duplicate_rows(
    validated_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Removes exact duplicate rows from the dataset.

    Args:
        validated_data: The schema-validated dataset.
        parameters: The ``data_processing`` parameters dictionary. May
            contain a ``duplicate_subset`` key listing the columns used to
            identify duplicates. If absent or ``None``, all columns are
            used (an exact full-row duplicate check).

    Returns:
        The dataset with duplicate rows dropped, index reset.
    """
    subset = parameters.get("duplicate_subset")
    n_before = len(validated_data)

    deduplicated = validated_data.drop_duplicates(subset=subset, keep="first")
    n_after = len(deduplicated)
    n_removed = n_before - n_after

    logger.info(
        "Duplicate removal: %d rows removed (%.2f%% of dataset). %d rows remain.",
        n_removed,
        (n_removed / n_before * 100) if n_before else 0.0,
        n_after,
    )

    return deduplicated.reset_index(drop=True)


def handle_missing_values(
    deduplicated_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Handles missing values according to a per-column strategy.

    The current historical dataset has zero missing values (confirmed
    during EDA), but this node exists so the pipeline behaves correctly
    and predictably if a future data refresh introduces gaps — it should
    never be skipped just because "the data is clean today."

    Args:
        deduplicated_data: The dataset after duplicate removal.
        parameters: The ``data_processing`` parameters dictionary. Must
            contain a ``missing_value_strategy`` mapping of column name to
            one of ``"drop_row"``, ``"median"``, ``"mean"``, or ``"mode"``.
            Columns not listed use ``"drop_row"`` as the default strategy.

    Returns:
        The dataset with missing values resolved according to the
        configured strategy.

    Raises:
        ValueError: If an unsupported strategy name is configured for a
            column.
    """
    strategy_map: dict[str, str] = parameters.get("missing_value_strategy", {})
    df = deduplicated_data.copy()

    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    if columns_with_missing.empty:
        logger.info("No missing values detected. Skipping imputation.")
        return df

    logger.info(
        "Missing values detected in %d column(s):\n%s",
        len(columns_with_missing),
        columns_with_missing.to_string(),
    )

    rows_to_drop_mask = pd.Series(False, index=df.index)

    for column in columns_with_missing.index:
        strategy = strategy_map.get(column, "drop_row")

        if strategy == "drop_row":
            rows_to_drop_mask |= df[column].isnull()
        elif strategy == "median":
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            logger.info("Column '%s': filled with median (%.4f).", column, fill_value)
        elif strategy == "mean":
            fill_value = df[column].mean()
            df[column] = df[column].fillna(fill_value)
            logger.info("Column '%s': filled with mean (%.4f).", column, fill_value)
        elif strategy == "mode":
            fill_value = df[column].mode(dropna=True).iloc[0]
            df[column] = df[column].fillna(fill_value)
            logger.info("Column '%s': filled with mode ('%s').", column, fill_value)
        else:
            raise ValueError(
                f"Unsupported missing_value_strategy '{strategy}' for column "
                f"'{column}'. Expected one of: drop_row, median, mean, mode."
            )

    n_before = len(df)
    df = df.loc[~rows_to_drop_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    if n_dropped:
        logger.info(
            "Dropped %d row(s) due to 'drop_row' strategy on one or more columns.",
            n_dropped,
        )

    return df


def validate_value_ranges(
    imputed_data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """Flags and removes rows with values outside sane business ranges.

    This guards against corrupt or nonsensical records (e.g. a negative
    ride duration, a rating outside 1-5) that could otherwise silently
    distort feature engineering and model training. Every row removed is
    recorded in a data quality report rather than dropped silently, so the
    action is auditable.

    Args:
        imputed_data: The dataset after missing value handling.
        parameters: The ``data_processing`` parameters dictionary. Must
            contain a ``value_ranges`` mapping of column name to a
            ``{"min": ..., "max": ...}`` dict describing the accepted
            inclusive range for that column. Columns not listed are not
            range-checked.

    Returns:
        A tuple of:
            * The dataset with out-of-range rows removed, index reset.
            * A data quality report dataframe summarising how many rows
              were removed per column and why, for auditing purposes.
    """
    value_ranges: dict[str, dict[str, float]] = parameters.get("value_ranges", {})
    df = imputed_data.copy()

    invalid_mask = pd.Series(False, index=df.index)
    report_rows = []

    for column, bounds in value_ranges.items():
        col_min, col_max = bounds["min"], bounds["max"]
        column_invalid_mask = (df[column] < col_min) | (df[column] > col_max)
        n_invalid = int(column_invalid_mask.sum())

        report_rows.append(
            {
                "column": column,
                "expected_min": col_min,
                "expected_max": col_max,
                "invalid_row_count": n_invalid,
                "invalid_row_pct": round(n_invalid / len(df) * 100, 2) if len(df) else 0.0,
            }
        )

        if n_invalid:
            logger.warning(
                "Column '%s': %d row(s) outside expected range [%s, %s].",
                column,
                n_invalid,
                col_min,
                col_max,
            )

        invalid_mask |= column_invalid_mask

    n_before = len(df)
    clean_df = df.loc[~invalid_mask].reset_index(drop=True)
    n_removed = n_before - len(clean_df)

    logger.info(
        "Value-range validation complete: %d row(s) removed (%.2f%% of dataset). "
        "%d rows remain.",
        n_removed,
        (n_removed / n_before * 100) if n_before else 0.0,
        len(clean_df),
    )

    quality_report = pd.DataFrame(report_rows)
    return clean_df, quality_report


def enforce_final_dtypes(
    clean_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:

    """Casts columns to their final production dtypes.

    Ensures downstream pipelines (feature engineering, training, inference)
    receive a dataset with guaranteed, consistent dtypes regardless of any
    upstream quirks (e.g. integers read as floats due to an unrelated
    missing value in the same column during a prior run).

    Args:
        clean_data: The fully validated and cleaned dataset.
        parameters: The ``data_processing`` parameters dictionary. Must
            contain a ``final_dtypes`` mapping of column name to the target
            pandas dtype string.

    Returns:
        The dataset with final dtypes enforced. This is the output
        consumed by the ``feature_engineering`` pipeline.
    """
    final_dtypes: dict[str, str] = parameters["final_dtypes"]
    df = clean_data.copy()

    for column, dtype in final_dtypes.items():
        try:
            df[column] = df[column].astype(dtype)
        except (ValueError, TypeError) as exc:
            logger.error(
                "Failed to cast column '%s' to dtype '%s': %s", column, dtype, exc
            )
            raise

    logger.info(
        "Final dtype enforcement complete for %d column(s). Output shape: %s.",
        len(final_dtypes),
        df.shape,
    )
    return df
