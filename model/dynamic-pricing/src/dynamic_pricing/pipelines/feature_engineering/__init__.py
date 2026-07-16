"""The ``feature_engineering`` pipeline.

Engineers the demand-supply ratio and encodes categorical features,
producing the model-ready feature table consumed by ``training`` and
``inference``. See ``README.md`` in this directory for full documentation.
"""

from .pipeline import create_feature_engineering_pipeline

__all__ = ["create_feature_engineering_pipeline"]
