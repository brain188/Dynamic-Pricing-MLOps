"""Integration smoke test: runs the full `data_processing` node chain in
sequence against a small synthetic dataset that mirrors the real schema,
including deliberately-injected issues (a duplicate row, a missing value,
an out-of-range rating) to confirm each node's cleanup actually fires
end-to-end, not just in isolated unit tests.
"""

import pandas as pd
import yaml

from dynamic_pricing.pipelines.data_processing.nodes import (
    enforce_final_dtypes,
    handle_missing_values,
    remove_duplicate_rows,
    validate_schema,
    validate_value_ranges,
)


def test_full_data_processing_chain_runs_end_to_end():
    with open("conf/base/parameters_data_processing.yml") as f:
        parameters = yaml.safe_load(f)["data_processing"]

    raw = pd.DataFrame(
        {
            "Number_of_Riders": pd.array([90, 58, 58, 42, 89, 70], dtype="int64"),
            "Number_of_Drivers": pd.array([45, 39, 39, 31, 28, 33], dtype="int64"),
            "Location_Category": ["Urban", "Suburban", "Suburban", "Rural", "Rural", "Urban"],
            "Customer_Loyalty_Status": ["Silver", "Silver", "Silver", "Silver", "Regular", "Gold"],
            "Number_of_Past_Rides": pd.array([10, 20, 20, 30, 40, 15], dtype="int64"),
            "Average_Ratings": pd.array([4.5, 3.9, 3.9, 9.9, 4.1, 4.0], dtype="float64"),  # 9.9 is invalid
            "Time_of_Booking": ["Night", "Morning", "Morning", "Evening", "Afternoon", "Night"],
            "Vehicle_Type": ["Premium", "Economy", "Economy", "Premium", "Economy", "Premium"],
            "Expected_Ride_Duration": pd.array([90, 60, 60, 45, 100, 80], dtype="int64"),
            "Historical_Cost_of_Ride": pd.array(
                [300.0, 250.0, 250.0, 220.0, 400.0, None], dtype="float64"
            ),  # last row has the missing target, kept separate from the invalid-rating row
        }
    )

    validated = validate_schema(raw, parameters)
    deduped = remove_duplicate_rows(validated, parameters)
    assert len(deduped) == 5, "the injected duplicate row should have been removed"

    imputed = handle_missing_values(deduped, parameters)
    assert imputed["Historical_Cost_of_Ride"].isnull().sum() == 0
    assert len(imputed) == 4, "the row with a missing target should be dropped, not imputed"

    range_checked, report = validate_value_ranges(imputed, parameters)
    assert (range_checked["Average_Ratings"] <= 5.0).all(), "the invalid 9.9 rating should be removed"
    assert report.loc[report["column"] == "Average_Ratings", "invalid_row_count"].iloc[0] == 1

    final = enforce_final_dtypes(range_checked, parameters)
    assert str(final["Location_Category"].dtype) == "category"
    assert str(final["Number_of_Riders"].dtype) == "int64"

    print("Full data_processing chain executed successfully end-to-end.")
    print(final)


if __name__ == "__main__":
    test_full_data_processing_chain_runs_end_to_end()
