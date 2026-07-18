"""Environment-driven configuration.

All runtime config comes from environment variables (12-factor style),
with sane local-dev defaults. No secrets or environment-specific values
are hardcoded elsewhere in the app.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DYNAMIC_PRICING_")

    # MLflow model resolution — mirrors model/conf/base/parameters.yml
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    registered_model_name: str = "dynamic_pricing_model"
    production_alias: str = "production"

    # Fitted OneHotEncoder artifact from feature_engineering. Loaded from
    # disk, not the MLflow registry, since it's a preprocessing artifact
    # rather than a model.
    encoder_path: str = "artifacts/categorical_encoder.pickle"

    # Feature engineering parameters — must match
    # model/conf/base/parameters.yml exactly, or the
    # backend will compute features differently than training did.
    demand_supply_numerator_column: str = "Number_of_Riders"
    demand_supply_denominator_column: str = "Number_of_Drivers"
    demand_supply_outlier_cap_percentile: float = 0.99
    categorical_columns: list[str] = [
        "Location_Category",
        "Customer_Loyalty_Status",
        "Time_of_Booking",
        "Vehicle_Type",
    ]

    prediction_min: float = 0
    prediction_max: float = 5000

    # Audit log: append-only JSONL, one record per request. This is the
    # file the monitoring pipeline (Evidently) reads to compute drift and
    # error-rate metrics, and what an ops dashboard tails for failures.
    audit_log_path: str = "logs/audit.jsonl"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — env vars are read once per process."""
    return Settings()
