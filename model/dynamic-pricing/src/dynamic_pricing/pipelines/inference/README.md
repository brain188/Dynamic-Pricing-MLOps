# Pipeline: `inference`

## Purpose

Scores new, unseen ride requests using whichever model is currently
aliased `"production"` in the MLflow Model Registry. This pipeline is the
final link in the chain: `data_processing` → `feature_engineering` →
`training` → **`inference`** — and it's built around one non-negotiable
rule: **a new ride request must be transformed exactly the way training
data was transformed, with zero drift between the two.**

## The core design decision: reuse, don't reimplement

This pipeline's `pipeline.py` imports and wires in these functions
**directly, unchanged**:

- `data_processing.nodes.validate_schema`
- `feature_engineering.nodes.engineer_demand_supply_ratio`
- `feature_engineering.nodes.encode_categorical_features`

It does **not** contain a second, parallel implementation of schema
checking or feature engineering. If either of those functions changes in
the future (a bug fix, a new business rule), inference picks up the
change automatically on the next run — there is no second copy that
could silently drift out of sync with training. This is what "consistent
features between training and production" (MLOps component #6 in the
project's architecture) actually means in code, not just in a diagram.

The categorical encoding step also reuses the **exact fitted
`categorical_encoder` object** persisted by `feature_engineering` — it is
loaded from `data/04_feature/categorical_encoder.pickle`, never re-fit.
Re-fitting a new encoder on inference-time data would silently break
column alignment with the trained model the moment category frequencies
differ even slightly.

## The model is loaded by alias, never by hardcoded version

`load_production_model` resolves `models:/{registered_model_name}@production`
— an alias, not a version number. This matters concretely for this
project: the production model has already changed once (the duration-only
baseline won initially, then a retuned CatBoost won after regularized
hyperparameter search). Promoting a future retraining winner is a
one-line MLflow registry operation; this pipeline's code never needs to
change to pick it up.

## Column selection is read from the model, not hardcoded

`generate_predictions` inspects the production model's own MLflow
signature (`model.metadata.get_input_schema()`) to determine which
columns to pass to `.predict()`. This deliberately avoids hardcoding
"the production model uses all features" — which would have been *wrong*
the first time this project trained a model (the baseline, using only
`Expected_Ride_Duration`, won initially) and will likely need to change
again after future retraining. If the model's required columns are ever
missing from the engineered inference data, this function raises
immediately rather than silently mispredicting — that specific failure
mode (training/serving feature skew) is exactly what this check exists
to catch.

## How to run it

```bash
kedro run --pipeline inference
```

Run the unit and integration tests for this pipeline:

```bash
pytest tests/pipelines/test_inference/
```

> The integration test trains a real (tiny) CatBoost model, fits a real
> `categorical_encoder`, registers the model in a temporary real MLflow
> SQLite store, aliases it `"production"`, and then runs the **entire**
> inference chain against it end-to-end — this is the test that actually
> proves the training/serving contract holds, not a mocked stand-in.

> **Known caveat, surfaced by a real MLflow warning during testing:**
> MLflow flags that integer input columns (e.g. `Number_of_Riders`) can't
> represent missing values in their inferred schema. If a live inference
> request ever arrives with a null in an integer column, schema
> enforcement may reject it. `data_processing`'s schema validation
> already guards against this for training data; the same discipline
> should extend to whatever validates real-time request payloads in the
> backend.

## Why `inference` is registered separately, not part of `__default__`

`data_processing` + `feature_engineering` + `training` form one coherent
batch job: given the historical dataset, produce a trained, registered
model. `inference` is a fundamentally different kind of run: given a
new, already-registered model and *new* incoming ride requests, produce
predictions. Chaining it into the same `__default__` pipeline would imply
"retrain, then immediately score" every single run, which isn't the
right operational shape — inference should run far more frequently than
training (potentially once per request, in the live-serving case) and
independently of it. It's registered as its own named pipeline instead:
`kedro run --pipeline=inference`.

## Inputs

| Dataset | Catalog entry | Source |
|---|---|---|
| New ride requests | `new_ride_requests` | `data/01_raw/new_ride_requests.csv` (batch/offline mode) |
| Fitted categorical encoder | `categorical_encoder` | Reused from `feature_engineering` (`data/04_feature/categorical_encoder.pickle`) |

Parameters are read from `params:inference` (defined in
`conf/base/parameters_inference.yml`) **and** `params:feature_engineering`
(reused directly, since the imported feature engineering functions expect
that exact parameter shape). Key `inference` parameter groups:

- `expected_schema` — same as `data_processing`'s, minus the target column
- `mlflow` — tracking URI, registered model name, and the alias to resolve
- `prediction_column_name` — output column name
- `sanity_bounds` — Requirement P4's min/max bounds and clip-vs-flag behavior

## Pipeline steps (nodes)

1. **`load_production_model_node`** — resolves and loads the
   `"production"`-aliased model from the MLflow registry.
2. **`validate_inference_schema_node`** — reused from `data_processing`;
   validates new request data against the (target-column-free) expected
   schema.
3. **`engineer_inference_demand_supply_ratio_node`** — reused from
   `feature_engineering`; computes and caps the demand-supply ratio
   identically to training.
4. **`encode_inference_categorical_features_node`** — reused from
   `feature_engineering`; applies the persisted, already-fitted one-hot
   encoder.
5. **`generate_predictions_node`** — selects the model's required input
   columns via its logged signature and generates predictions.
6. **`validate_predictions_node`** — enforces Requirement P4's sanity
   bounds (clips or flags out-of-range predictions).

## Outputs

| Dataset | Catalog entry | Description |
|---|---|---|
| Final predictions | `predictions_output` | Engineered request data + `predicted_fare` + `is_out_of_bounds` flag. |

Intermediate outputs (`validated_ride_requests`, `ride_requests_with_ratio`,
`encoded_ride_requests`, `raw_predictions`) are persisted to
`data/02_intermediate/` for inspection.

## What happens next: the serving layer

This pipeline covers **batch** inference (a file of new requests in,
predictions out). The FastAPI backend reuses this exact
logic for **real-time, single-request** inference: it will call
`mlflow.pyfunc.load_model(f"models:/{registered_model_name}@production")`
directly (mirroring `load_production_model`), apply the same
`categorical_encoder` and `engineer_demand_supply_ratio` logic to a
single incoming request, and return one prediction synchronously — the
same reuse principle this pipeline establishes, just behind an API
instead of a `kedro run`.
