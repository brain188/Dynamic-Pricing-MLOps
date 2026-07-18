"""Single-request feature engineering.

Reuses `engineer_demand_supply_ratio` and `encode_categorical_features`
directly from `dynamic_pricing.pipelines.feature_engineering.nodes` — the
exact functions `training` and the Kedro `inference` pipeline use — so a
live request is transformed identically to how training data was. This
module's only job is adapting "one Pydantic request" into "one-row
DataFrame in, encoded DataFrame out"; it contains no feature logic of its
own.
"""

import pandas as pd
from dynamic_pricing.pipelines.feature_engineering.nodes import (
    encode_categorical_features,
    engineer_demand_supply_ratio,
)

from .model_loader import ModelBundle
from .schemas import RideRequest


def transform_request(request: RideRequest, bundle: ModelBundle, settings) -> pd.DataFrame:
    """Turns one validated ride request into one row of model-ready features.

    Args:
        request: The validated incoming request.
        bundle: Holds the fitted encoder used for categorical encoding.
        settings: Application settings, providing the exact
            feature-engineering parameters used at training time.

    Returns:
        A single-row dataframe with the same columns/encoding the
        production model was trained on.

    Note:
        `engineer_demand_supply_ratio`'s outlier cap is computed as a
        percentile of whatever batch it's given. For a single-row
        request, that percentile is just the row's own value, so no
        capping can occur here — capping is only meaningful across a
        batch. This is a known, accepted limitation of scoring one
        request at a time: an extreme single request (e.g. 1 driver,
        100 riders) reaches the model uncapped. If this becomes a
        problem in practice, the fix is to cap against the *training-time*
        cap value (persisted as an artifact) instead of recomputing a
        percentile per request — not implemented here to avoid adding an
        artifact this project doesn't otherwise need.
    """
    raw_row = pd.DataFrame([request.model_dump()])

    fe_params = {
        "demand_supply_ratio": {
            "numerator_column": settings.demand_supply_numerator_column,
            "denominator_column": settings.demand_supply_denominator_column,
            "outlier_cap_percentile": settings.demand_supply_outlier_cap_percentile,
        },
        "categorical_columns": settings.categorical_columns,
    }

    with_ratio = engineer_demand_supply_ratio(raw_row, fe_params)
    encoded = encode_categorical_features(with_ratio, bundle.encoder, fe_params)

    return encoded
