# Pipeline: `feature_engineering`

## Purpose

Turns the clean dataset from `data_processing` into a fully numeric,
model-ready feature table. Two things happen here, and only here:

1. **Demand-supply ratio engineering** — the central hypothesis of this
   project (`Number_of_Riders ÷ Number_of_Drivers`), with outlier capping.
2. **Categorical encoding** — one-hot encoding via a *fitted, persisted*
   `sklearn.preprocessing.OneHotEncoder`, not `pandas.get_dummies`.

This pipeline directly implements Requirement **F2** ("model incorporates
a demand-supply ratio as a derived feature") and lays the groundwork for
**F3** ("pricing responds to location, loyalty, and time of booking") by
making those categoricals model-consumable.

## Why a fitted, persisted encoder instead of `pd.get_dummies`

This is the most important design decision in this pipeline, worth
understanding before touching the code:

`pandas.get_dummies` infers which columns to create from whatever data
it's given *at that moment*. If training data has 3 `Location_Category`
values but a future inference request's data only contains 2 of them
(or a new value show up), `get_dummies` would silently produce a
different, mismatched set of columns — breaking the model at inference
time in a way that's easy to miss until it fails in production.

Instead, this pipeline:

- **Fits** a `OneHotEncoder` once, on the full feature-engineering dataset
  (`fit_categorical_encoder_node`), learning a fixed set of known categories.
- **Persists** that fitted encoder as its own catalog artifact
  (`categorical_encoder`), stored under `data/04_feature/` — this project's
  lightweight feature store layer (MLOps component #6).
- **Transforms** using that exact fitted encoder (`encode_categorical_features_node`),
  with `handle_unknown="ignore"` so any category not seen during fitting
  is safely encoded as all-zeros instead of crashing.

The `inference` pipeline (and later, the FastAPI backend) will **load
this same persisted encoder** rather than fitting a new one — this is
what guarantees training and serving compute features identically
(MLOps component: consistent train/serve features).

## How to run it

```bash
kedro run --pipeline feature_engineering
```

Run the unit and integration tests for this pipeline:

```bash
pytest tests/pipelines/test_feature_engineering/
```

## Inputs

| Dataset | Catalog entry | Source |
|---|---|---|
| Cleaned ride data | `cleaned_ride_data` | Output of `data_processing` (`data/03_primary/cleaned_ride_data.parquet`) |

Parameters are read from `params:feature_engineering`, defined in
`conf/base/parameters_feature_engineering.yml`. Key parameter groups:

- `demand_supply_ratio` — numerator/denominator columns and the outlier
  cap percentile (0.99, derived from the EDA outlier check)
- `categorical_columns` — the four categoricals to encode
- `encoder` — `handle_unknown` and `drop` settings for the `OneHotEncoder`
- `target_column` — kept so the final feature table places it last

## Pipeline steps (nodes)

Executed in this order:

1. **`engineer_demand_supply_ratio_node`** — computes the ratio and caps
   values above the configured percentile (winsorizing rather than
   dropping rows, to avoid losing otherwise-valid data over one feature).
2. **`fit_categorical_encoder_node`** — fits a `OneHotEncoder` on the four
   categorical columns and returns it as a standalone artifact.
3. **`encode_categorical_features_node`** — applies the fitted encoder,
   replacing the original categorical columns with their one-hot
   equivalents; all other columns pass through unchanged.
4. **`assemble_feature_table_node`** — final housekeeping: places the
   target column last, and raises loudly if any unexpected nulls were
   introduced during encoding (a bug signal, not an expected condition).

## Outputs

| Dataset | Catalog entry | Description |
|---|---|---|
| Feature table | `feature_table` | The final, fully numeric, model-ready table. **This is the contract handed to `training`.** |
| Categorical encoder | `categorical_encoder` | The fitted `OneHotEncoder`, persisted for reuse by `inference` and the backend. |

Intermediate outputs (`data_with_ratio`, `encoded_ride_data`) are
persisted to `data/02_intermediate/` for inspection and are not meant to
be consumed by other pipelines.

## Key findings from EDA/permutation importance that shaped this design

- The demand-supply ratio showed a weak, even slightly *negative* linear
  correlation with cost (r = -0.094), but ranked *above* its raw
  component columns in both impurity-based and permutation feature
  importance — evidence it captures a real (if modest) non-linear signal
  worth keeping, not evidence to drop it.
- Permutation importance corrected an impurity-based bias: `Vehicle_Type`
  ranked far higher (2nd overall) once measured fairly, confirming it's
  worth encoding carefully rather than treating as a minor feature.
- `Location_Category`, `Time_of_Booking`, and `Customer_Loyalty_Status`
  remain weak signals under both importance methods — kept anyway per
  Requirement F3, at negligible modeling cost.

## What happens next: `training`

The `training` pipeline consumes `feature_table` and is responsible for:

- Splitting into train/test sets.
- Training and comparing three candidate models: a duration-only Linear
  Regression baseline, a Random Forest, and a gradient
  boosting model (XGBoost or LightGBM).
- Logging all experiments, metrics (RMSE/MAE/R²), and model artifacts to
  MLflow (Experiment Tracking + Model Registry, MLOps components #7-8).
- Selecting and registering the winning model as "Production" in the
  MLflow Model Registry.

See `src/dynamic_pricing/pipelines/training/README.md` for
full details.
