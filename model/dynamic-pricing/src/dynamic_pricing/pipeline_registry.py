from kedro.pipeline import Pipeline

from .pipelines.data_processing import create_data_processing_pipeline as dp


def register_pipelines() -> dict[str, Pipeline]:
    """Registers the project's pipelines.

    """
    data_processing_pipeline = dp()

    return {
        "__default__": data_processing_pipeline,
        "data_processing": data_processing_pipeline,
    }
