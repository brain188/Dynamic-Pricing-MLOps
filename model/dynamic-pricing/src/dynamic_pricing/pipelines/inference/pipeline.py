from kedro.pipeline import Pipeline, node

from dynamic_pricing.pipelines.data_processing.nodes import validate_schema
from dynamic_pricing.pipelines.feature_engineering.nodes import (
    encode_categorical_features,
    engineer_demand_supply_ratio,
)

from .nodes import generate_predictions, load_production_model, validate_predictions


def create_inference_pipeline(**kwargs) -> Pipeline:
    """Creates the ``inference`` pipeline.

    """
    return Pipeline(
        [
            node(
                func    = load_production_model,
                inputs  = "params:inference",
                outputs = "production_model",
                name    = "load_production_model_node",
            ),
            node(
                func    = validate_schema,
                inputs  = ["new_ride_requests", "params:inference"],
                outputs = "validated_ride_requests",
                name    = "validate_inference_schema_node",
            ),
            node(
                func    = engineer_demand_supply_ratio,
                inputs  = ["validated_ride_requests", "params:feature_engineering"],
                outputs = "ride_requests_with_ratio",
                name    = "engineer_inference_demand_supply_ratio_node",
            ),
            node(
                func    = encode_categorical_features,
                inputs  = [
                    "ride_requests_with_ratio",
                    "categorical_encoder",
                    "params:feature_engineering",
                ],
                outputs = "encoded_ride_requests",
                name    = "encode_inference_categorical_features_node",
            ),
            node(
                func    = generate_predictions,
                inputs  = ["encoded_ride_requests", "production_model", "params:inference"],
                outputs = "raw_predictions",
                name    = "generate_predictions_node",
            ),
            node(
                func    = validate_predictions,
                inputs  = ["raw_predictions", "params:inference"],
                outputs = "predictions_output",
                name    = "validate_predictions_node",
            ),
        ]
    )
