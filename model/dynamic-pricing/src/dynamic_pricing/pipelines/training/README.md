# Pipeline: `training`

## Purpose

Trains and compares five regression models against the feature table from
`feature_engineering`, logs every one to MLflow, and promotes the best
performer to the MLflow Model Registry under a `"production"` alias. This
pipeline directly implements Requirements **P1** and **P2**: accuracy is
quantified per model, and every candidate is measured against the
mandatory duration-only baseline.

## The five models

| # | Model | Features used | Role |
|---|---|---|---|
| 1 | Linear Regression | `Expected_Ride_Duration` only | Mandatory baseline — mirrors the company's *current* pricing approach |
| 2 | Random Forest | All features | Interpretable, robust "safe middle" candidate |
| 3 | LightGBM | All features | Gradient boosting candidate |
| 4 | XGBoost | All features | Gradient boosting candidate |
| 5 | CatBoost | All features | Gradient boosting candidate |

All three gradient boosting libraries are trained and compared rather
than picking one upfront — which one wins is an empirical question this
pipeline answers with evidence (the comparison table), not a guess made
in advance.

## Why every model gets registered, not just the winner

Every one of the five models is logged as its own MLflow run **and**
registered as its own version under the same `registered_model_name`.
This is intentional, not wasteful: the registry's job is to hold every
candidate that was seriously evaluated, with the `"production"` alias
marking which one is currently promoted for serving. This means:

- You can always answer "what did we try, and how did each one do?"
  directly from the registry, months later, without re-running training.
- Promoting a *different* registered version later (e.g. after
  retraining finds a new winner) is a one-line alias change
  (`client.set_registered_model_alias(...)`), not a retraining run.

## MLflow tracking backend — important version note

MLflow **3.0+** puts the legacy filesystem tracking store (a bare
`./mlruns` folder) into maintenance mode and will refuse to write to one
by default. This pipeline's `mlflow.tracking_uri` parameter therefore
points at a local SQLite database (`sqlite:///mlflow.db`) instead. If
you're running an older MLflow version, a bare folder path still works,
but the SQLite URI works on both — no reason to change it back.

To inspect results in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## How to run it

```bash
kedro run --pipeline training
```

Run the unit and integration tests for this pipeline:

```bash
pytest tests/pipelines/test_training/
```

## Inputs

| Dataset | Catalog entry | Source |
|---|---|---|
| Feature table | `feature_table` | Output of `feature_engineering` (`data/04_feature/feature_table.parquet`) |

Parameters are read from `params:training`, defined in
`conf/base/parameters_training.yml`. Key parameter groups:

- `target_column`, `test_size`, `random_state` — train/test split config
- `baseline_model_key`, `baseline_feature_columns` — which model is "the
  baseline" and which column(s) it's restricted to
- `random_forest` / `lightgbm` / `xgboost` / `catboost` — per-model
  hyperparameters (currently modest, default-leaning values; a tuning
  pass is a natural next step once the winning algorithm is known)
- `mlflow` — tracking URI, experiment name, registered model name, and
  the alias used to mark the production model

## Pipeline steps (nodes)

1. **`split_data_node`** — train/test split (80/20 by default).
2. **`train_baseline_model_node`**, **`train_random_forest_node`**,
   **`train_lightgbm_model_node`**, **`train_xgboost_model_node`**,
   **`train_catboost_model_node`** — five independent training nodes with
   no dependency on each other, so Kedro's `ParallelRunner` can run them
   concurrently.
3. **`collect_trained_models_node`** — joins all five into a single named
   dictionary for the evaluation/logging steps.
4. **`evaluate_models_node`** — computes RMSE/MAE/R² for every model on
   the held-out test set (the baseline is evaluated only on its
   configured column subset) and returns a comparison table sorted by
   RMSE ascending.
5. **`log_and_register_best_model_node`** — logs every model to MLflow
   with its metrics and parameters, registers each as a new model
   version, and assigns the `"production"` alias to the version with the
   best RMSE.

## Outputs

| Dataset | Catalog entry | Description |
|---|---|---|
| Model comparison table | `model_comparison_table` | RMSE/MAE/R² per model, sorted best-first. Your evidence for Requirements P1/P2. |
| Training summary | `training_summary` | JSON: which model won, its registry version, and its test metrics. |
| Individual model artifacts | `baseline_model`, `random_forest_model`, `lightgbm_model`, `xgboost_model`, `catboost_model` | Persisted for direct inspection outside MLflow. |
| Registered models | *(in MLflow, not the Kedro catalog)* | Five versions under `registered_model_name`, with `"production"` alias on the winner. |

## What happens next: `inference`

The `inference` pipeline (and later, the FastAPI backend) will:

- Load the model via `mlflow.pyfunc.load_model(f"models:/{registered_model_name}@production")` —
  resolving by alias, never by hardcoded version number, so promoting a
  new winner later requires no code change.
- Load the persisted `categorical_encoder` from `feature_engineering` to
  apply identical preprocessing to new, unseen ride requests.
- Reuse the `engineer_demand_supply_ratio` logic from
  `feature_engineering` so a live prediction request is transformed
  exactly like training data was.

See `src/dynamic_pricing/pipelines/inference/README.md` for full details.
