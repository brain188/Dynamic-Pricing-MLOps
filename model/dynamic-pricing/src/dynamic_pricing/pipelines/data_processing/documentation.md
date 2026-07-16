# Pipeline: `data_processing`

## Purpose

Turns the raw historical ride-sharing dataset into a clean, schema-validated,
range-checked dataset with correct dtypes — nothing more. This pipeline does
**not** perform feature engineering (no demand-supply ratio, no encoding
beyond dtype casting to `category`). That separation is intentional: this
pipeline can be reused unchanged even if the feature engineering strategy
changes later, and vice versa.

This pipeline directly implements Requirement **D1** from the project
charter: *"No unresolved nulls or duplicates in the final modeling
dataset."*

## How to run it

Run just this pipeline in isolation:

```bash
kedro run --pipeline data_processing
```

Run it as part of the full project (once other pipelines exist):

```bash
kedro run
```

Visualize the pipeline graph:

```bash
kedro viz
```

Run the unit tests for this pipeline:

```bash
pytest tests/pipelines/test_data_processing/
```

## Inputs

| Dataset | Catalog entry | Source |
|---|---|---|
| Raw historical ride data | `raw_ride_data` | `data/01_raw/dynamic_pricing.csv` |

Parameters are read from `params:data_processing`, defined in
`conf/base/parameters_data_processing.yml`. Key parameter groups:

- `expected_schema` — the columns and dtypes the pipeline asserts on entry
- `duplicate_subset` — which columns define a "duplicate" row
- `missing_value_strategy` — per-column imputation/drop rule
- `value_ranges` — sane min/max bounds per column
- `final_dtypes` — the dtypes guaranteed on the pipeline's output

## Pipeline steps (nodes)

Executed in this order:

1. **`validate_schema_node`** — asserts all expected columns are present
   with the expected dtypes. Raises immediately on mismatch; this is a
   fail-fast guardrail against upstream schema drift.
2. **`remove_duplicate_rows_node`** — drops exact duplicate rows.
3. **`handle_missing_values_node`** — resolves missing values per the
   configured per-column strategy (median / mean / mode / drop_row).
   Currently a no-op safety net, since the historical dataset has zero
   missing values (confirmed in `01_eda.ipynb`), but must not be removed
   — future data refreshes are not guaranteed to stay clean.
4. **`validate_value_ranges_node`** — removes rows with values outside
   sane business bounds (e.g. a rating outside 1–5) and produces an
   auditable data quality report of what was removed and why.
5. **`enforce_final_dtypes_node`** — casts every column to its final,
   guaranteed production dtype (categoricals become pandas `category`
   dtype here, ready for consistent downstream encoding).

## Outputs

| Dataset | Catalog entry | Description |
|---|---|---|
| Cleaned ride data | `cleaned_ride_data` | The validated, deduplicated, range-checked dataset with final dtypes. **This is the contract handed to `feature_engineering`.** |
| Data quality report | `data_quality_report` | Auditable summary of how many rows were removed per column during range validation, and why. |

Intermediate per-node outputs (`validated_ride_data`,
`deduplicated_ride_data`, `imputed_ride_data`, `range_validated_ride_data`)
are also persisted to `data/02_intermediate/` so each step can be
inspected or re-run independently — they are not meant to be consumed by
other pipelines.

## Key findings from EDA that shaped this pipeline's design

From `01_eda.ipynb`:

- Zero missing values and zero duplicates in the historical dataset —
  meaning this pipeline's imputation/deduplication logic is currently a
  safety net rather than an active fix. Do not assume future data will
  stay this clean.
- No extreme outliers in the core numeric columns (0–1% via IQR), except
  the *engineered* `demand_supply_ratio` (7% outliers) — that concern
  belongs to `feature_engineering`, not here, since the ratio doesn't
  exist yet at this stage of the pipeline.

## What happens next: `feature_engineering`

The `feature_engineering` pipeline consumes `cleaned_ride_data` and is
responsible for:

- Engineering the `demand_supply_ratio` feature (`Number_of_Riders` ÷
  `Number_of_Drivers`), including the outlier-capping logic identified as
  necessary during EDA (7% of ratio values were outliers by the IQR
  method).
- Encoding categorical features (`Location_Category`,
  `Customer_Loyalty_Status`, `Time_of_Booking`, `Vehicle_Type`) for
  modeling.
- Writing the resulting feature table to `data/04_feature/`, which acts
  as this project's lightweight feature store.

See `src/dynamic_pricing/pipelines/feature_engineering/README.md` (once
created) for full details.
