"""The ``training`` pipeline.

Trains and compares five regression models (Linear Regression baseline,
Random Forest, LightGBM, XGBoost, CatBoost), logs every run to MLflow, and
promotes the best model to the ``"production"`` registry alias. See
``README.md`` in this directory for full documentation.
"""

from .pipeline import create_training_pipeline

__all__ = ["create_training_pipeline"]
