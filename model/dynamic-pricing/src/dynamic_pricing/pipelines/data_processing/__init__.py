"""The ``data_processing`` pipeline.

Cleans and validates raw ride-sharing data before it is handed to the
``feature_engineering`` pipeline. See ``README.md`` in this directory for
full documentation.
"""

from .pipeline import create_data_processing_pipeline

__all__ = ["create_data_processing_pipeline"]
