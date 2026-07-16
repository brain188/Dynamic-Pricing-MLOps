from kedro.pipeline import Pipeline, node

from .nodes import (
    assemble_feature_table,
    encode_categorical_features,
    engineer_demand_supply_ratio,
    fit_categorical_encoder,
)


def create_feature_engineering_pipeline(**kwargs) -> Pipeline:
    """Creates the ``feature_engineering`` pipeline.

    """
    return Pipeline(
        [
            node(
                func    = engineer_demand_supply_ratio,
                inputs  = ["cleaned_ride_data", "params:feature_engineering"],
                outputs = "data_with_ratio",
                name    = "engineer_demand_supply_ratio_node",
            ),
            node(
                func    = fit_categorical_encoder,
                inputs  = ["data_with_ratio", "params:feature_engineering"],
                outputs = "categorical_encoder",
                name    = "fit_categorical_encoder_node",
            ),
            node(
                func    = encode_categorical_features,
                inputs  = [
                    "data_with_ratio",
                    "categorical_encoder",
                    "params:feature_engineering",
                ],
                outputs = "encoded_ride_data",
                name    = "encode_categorical_features_node",
            ),
            node(
                func    = assemble_feature_table,
                inputs  = ["encoded_ride_data", "params:feature_engineering"],
                outputs = "feature_table",
                name    = "assemble_feature_table_node",
            ),
        ]
    )
