"""The ``inference`` pipeline.

Scores new ride requests with the MLflow-registered model currently
aliased ``"production"``, reusing the exact schema validation and feature
engineering node functions used at training time. See ``README.md`` in
this directory for full documentation.
"""

from .pipeline import create_inference_pipeline

__all__ = ["create_inference_pipeline"]
