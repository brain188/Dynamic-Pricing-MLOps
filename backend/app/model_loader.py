"""Model and encoder loading.

Reuses `dynamic_pricing.pipelines.inference.nodes.load_production_model`
directly — the exact same function the Kedro `inference` pipeline uses —
so "how the backend resolves a model" and "how batch inference resolves
a model" can never silently diverge into two implementations.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
from dynamic_pricing.pipelines.inference.nodes import load_production_model
from sklearn.preprocessing import OneHotEncoder

from .config import Settings

logger = logging.getLogger(__name__)


class ModelBundle:
    """Holds everything needed to serve a prediction: the model, the
    fitted encoder, and the model's registry version for audit trails."""

    def __init__(
        self,
        model: mlflow.pyfunc.PyFuncModel,
        encoder: OneHotEncoder,
        model_version: str,
    ):
        self.model = model
        self.encoder = encoder
        self.model_version = model_version


def load_model_bundle(settings: Settings) -> ModelBundle:
    """Loads the production model and its paired categorical encoder.

    Args:
        settings: Application settings with MLflow and encoder-path config.

    Returns:
        A populated `ModelBundle`.

    Raises:
        FileNotFoundError: If the encoder artifact is missing.
        Exception: Any MLflow error is left unwrapped and propagates —
            the caller (app startup) treats this as fatal, since serving
            without a model is not a valid degraded state.
    """
    inference_params = {
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "registered_model_name": settings.registered_model_name,
            "production_alias": settings.production_alias,
        }
    }
    model = load_production_model(inference_params)
    model_version = _resolve_model_version(settings)

    encoder_path = Path(settings.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Categorical encoder not found at '{encoder_path}'. This must be "
            "the exact fitted encoder produced by the feature_engineering "
            "pipeline — copy data/04_feature/categorical_encoder.pickle from "
            "the model/ project."
        )
    with encoder_path.open("rb") as f:
        encoder = pickle.load(f)

    logger.info(
        "Loaded model bundle: version=%s, encoder=%s.",
        model_version,
        encoder_path,
    )
    return ModelBundle(model=model, encoder=encoder, model_version=model_version)


def _resolve_model_version(settings: Settings) -> Optional[str]:
    """Resolves the registry version number behind the production alias,
    for inclusion in every audit record and API response.

    MLflow returns `version` as an int; this is coerced to `str` here,
    once, at the source — so every downstream consumer (audit records,
    API responses) can rely on it always being a string without each one
    needing to remember to convert it.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(
        settings.registered_model_name, settings.production_alias
    )
    return str(version.version)
