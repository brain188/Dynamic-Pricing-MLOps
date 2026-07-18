# `backend` — Dynamic Pricing FastAPI Service

## Purpose

Serves real-time fare predictions from the model currently aliased
`"production"` in the MLflow Model Registry (populated by `model/`'s
`training` pipeline). This is the live-serving counterpart to `model/`'s
batch `inference` pipeline — same underlying logic, different entry
point (an HTTP request instead of a `kedro run`).

## The core architectural decision: no reimplementation

This backend has **zero duplicated feature-engineering or model-loading
logic**. It depends on the `model/` package directly (`dynamic-pricing-model`,
installed as a local path dependency) and imports its node functions:

| Backend module | Reuses |
|---|---|
| `app/model_loader.py` | `dynamic_pricing.pipelines.inference.nodes.load_production_model` |
| `app/feature_pipeline.py` | `dynamic_pricing.pipelines.feature_engineering.nodes.engineer_demand_supply_ratio`, `encode_categorical_features` |

This is what actually guarantees training/serving consistency — not a
comment promising the two are "kept in sync," but the literal same
Python functions running in both places. If `feature_engineering`'s
ratio-capping logic changes, this backend picks up the change on its
next deploy with zero code edits here.

**Dependency footprint stays lean anyway**: the backend depends on
`dynamic-pricing-model`'s *core* install only (pandas, scikit-learn,
mlflow-skinny) — not its `[kedro]` or `[training]` extras. Kedro, XGBoost,
LightGBM, and full MLflow never ship in this service's image. See
`model/pyproject.toml` for how that split is enforced.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + whether a model is currently loaded |
| `POST` | `/predict` | Score one ride request, returns a fare prediction |

Interactive API docs (Swagger UI) are available at `/docs` once running.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Number_of_Riders": 90,
    "Number_of_Drivers": 45,
    "Location_Category": "Urban",
    "Customer_Loyalty_Status": "Silver",
    "Number_of_Past_Rides": 10,
    "Average_Ratings": 4.5,
    "Time_of_Booking": "Night",
    "Vehicle_Type": "Premium",
    "Expected_Ride_Duration": 90
  }'
```

```json
{
  "request_id": "3178af8d-9bb1-4ba7-9fd1-a8f813973457",
  "predicted_fare": 318.42,
  "is_out_of_bounds": false,
  "model_version": "1",
  "latency_ms": 7.0
}
```

## Auditing and monitoring — how failures become visible

Every request, successful or not, is written as one JSON line to the
audit log (`app/audit.py`, path configured via
`DYNAMIC_PRICING_AUDIT_LOG_PATH`):

```json
{"timestamp": "...", "request_id": "...", "event_type": "prediction", "status": "success", "input": {...}, "prediction": 318.42, "model_version": "1", "latency_ms": 7.0, "error": null}
{"timestamp": "...", "request_id": "...", "event_type": "prediction", "status": "failure", "input": null, "prediction": null, "model_version": null, "latency_ms": null, "error": "..."}
```

This is deliberately the **same flat JSONL shape** a monitoring tool
(e.g. `pd.read_json(path, lines=True)` feeding an Evidently report) can
consume with zero preprocessing. Concretely, this file is what:

- The `data_and_model_monitoring` MLOps component (not yet built) reads
  to compute prediction drift, error rates, and latency percentiles over
  time.
- An ops dashboard tails to alert on `status == "failure"` in near real
  time.
- Gives full traceability: every prediction is linked to the exact input
  and exact `model_version` that produced it — essential for tracing a
  bad prediction back to a specific (possibly since-replaced) model
  after a future retrain.

Separately, **structured JSON application logs** go to stdout (see
`app/logging_config.py`) for container log aggregation (CloudWatch,
Datadog, Loki) — this is operational/debugging visibility, distinct from
the audit log's business-record purpose. Both carry the same
`request_id` so a single request can be traced across both.

## Error handling

| Failure | Response | Audited? |
|---|---|---|
| Invalid request body (e.g. bad `Vehicle_Type`, `Number_of_Drivers <= 0`) | `422`, structured `ErrorResponse` | Yes, `status: "failure"` |
| Any unhandled exception (model error, feature engineering bug, etc.) | `500`, generic message (no internals leaked to the caller) | Yes, `status: "failure"`, full traceback in application logs |
| Model fails to load at startup | Process exits — a pricing service with no model is not a valid "degraded" state to serve from | Yes, `event_type: "startup"` |

## Configuration

All config is environment-driven (`app/config.py`), prefixed
`DYNAMIC_PRICING_`:

| Variable | Purpose |
|---|---|
| `DYNAMIC_PRICING_MLFLOW_TRACKING_URI` | Same registry `training` wrote to |
| `DYNAMIC_PRICING_REGISTERED_MODEL_NAME` | Same registered model name |
| `DYNAMIC_PRICING_PRODUCTION_ALIAS` | Which alias to resolve (default `production`) |
| `DYNAMIC_PRICING_ENCODER_PATH` | Path to the persisted `categorical_encoder.pickle` from `feature_engineering` |
| `DYNAMIC_PRICING_AUDIT_LOG_PATH` | Where audit records are appended |
| `DYNAMIC_PRICING_LOG_LEVEL` | Application log level |

## How to run it

```bash
cd backend
uv sync
export DYNAMIC_PRICING_MLFLOW_TRACKING_URI="sqlite:///../model/mlflow.db"
export DYNAMIC_PRICING_ENCODER_PATH="../model/data/04_feature/categorical_encoder.pickle"
uvicorn app.main:app --reload
```

Run the tests:

```bash
pytest tests/
```

> `tests/test_main.py` trains a real (tiny) CatBoost model, fits a real
> encoder, registers both in a temporary MLflow store, and drives the
> **actual FastAPI app** through `TestClient` — not mocks. This is what
> caught a real bug during development: MLflow's `ModelVersion.version`
> is an `int`, but `HealthResponse.model_version` was typed `str`,
> which crashed `/health` at runtime. The fix was coercing to `str`
> once, at the source (`model_loader._resolve_model_version`), rather
> than at every call site.

## What's deliberately not here

- **No retry/circuit-breaker logic against MLflow** — the model is
  loaded once at startup and held in memory (`app.state.model_bundle`)
  for the life of the process; a registry outage after startup doesn't
  affect already-running predictions. Restarting the process is how a
  newly-promoted model gets picked up (a common, deliberate pattern —
  the alternative, checking the registry on every request, would add
  latency to every single prediction for a rarely-changing value).
- **No authentication** — out of scope for this project; a real
  deployment would add an API gateway or middleware layer in front of
  this service.
