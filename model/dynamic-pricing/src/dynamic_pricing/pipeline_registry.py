from kedro.pipeline import Pipeline

from .pipelines.data_processing import create_data_processing_pipeline as dp
from .pipelines.feature_engineering import create_feature_engineering_pipeline as fe
from .pipelines.training import create_training_pipeline as tr


def register_pipelines() -> dict[str, Pipeline]:
    """Registers the project's pipelines.

    """
    data_processing_pipeline = dp()
    feature_engineering_pipeline = fe()
    training_pipeline = tr()

    return {
        "__default__": ( data_processing_pipeline
                         + feature_engineering_pipeline
                         + training_pipeline
        ),
        "data_processing": data_processing_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "training": training_pipeline
    }
