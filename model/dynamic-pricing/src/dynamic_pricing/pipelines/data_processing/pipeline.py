from kedro.pipeline import Pipeline, node

from .nodes import (
    enforce_final_dtypes,
    handle_missing_values,
    remove_duplicate_rows,
    validate_schema,
    validate_value_ranges,
)


def create_data_processing_pipeline(**kwargs) -> Pipeline:
    """Creates the ``data_processing`` pipeline.

    Returns:
        A Kedro ``Pipeline`` object with five nodes, executed in sequence:
        schema validation, duplicate removal, missing value handling,
        value-range validation, and final dtype enforcement.
    """
    return Pipeline(
        [
            node(
                func    = validate_schema,
                inputs  = ["raw_ride_data", "params:data_processing"],
                outputs = "validated_ride_data",
                name    = "validate_schema_node",
            ),
            node(
                func    = remove_duplicate_rows,
                inputs  = ["validated_ride_data", "params:data_processing"],
                outputs = "deduplicated_ride_data",
                name    = "remove_duplicate_rows_node",
            ),
            node(
                func    = handle_missing_values,
                inputs  = ["deduplicated_ride_data", "params:data_processing"],
                outputs = "imputed_ride_data",
                name    = "handle_missing_values_node",
            ),
            node(
                func    = validate_value_ranges,
                inputs  = ["imputed_ride_data", "params:data_processing"],
                outputs = ["range_validated_ride_data", "data_quality_report"],
                name    = "validate_value_ranges_node",
            ),
            node(
                func    = enforce_final_dtypes,
                inputs  = ["range_validated_ride_data", "params:data_processing"],
                outputs = "cleaned_ride_data",
                name    = "enforce_final_dtypes_node",
            ),
        ]
    )
