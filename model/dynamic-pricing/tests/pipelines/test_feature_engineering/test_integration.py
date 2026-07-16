"""Integration smoke test: runs the full `feature_engineering` node chain
in sequence, using the output shape/columns of `data_processing`'s smoke
test as a stand-in for `cleaned_ride_data`, to confirm both pipelines
connect correctly end-to-end.
"""

import pandas as pd
import yaml

from dynamic_pricing.pipelines.feature_engineering.nodes import (
    assemble_feature_table,
    encode_categorical_features,
    engineer_demand_supply_ratio,
    fit_categorical_encoder,
)


def test_full_feature_engineering_chain_runs_end_to_end():
    with open("conf/base/parameters_feature_engineering.yml") as f:
        parameters = yaml.safe_load(f)["feature_engineering"]

    # Shape mirrors what data_processing's cleaned_ride_data output looks like.
    cleaned_data = pd.DataFrame(
        {
            "Number_of_Riders": [90, 58, 42, 89, 70, 33, 12, 95],
            "Number_of_Drivers": [45, 39, 31, 3, 33, 20, 11, 40],  # row 3 -> high ratio outlier
            "Location_Category": [
                "Urban", "Suburban", "Rural", "Rural",
                "Urban", "Suburban", "Rural", "Urban",
            ],
            "Customer_Loyalty_Status": [
                "Silver", "Silver", "Regular", "Gold",
                "Gold", "Regular", "Silver", "Gold",
            ],
            "Number_of_Past_Rides": [10, 20, 30, 40, 15, 5, 60, 25],
            "Average_Ratings": [4.5, 3.9, 4.1, 4.8, 4.0, 3.7, 4.9, 4.2],
            "Time_of_Booking": [
                "Night", "Morning", "Evening", "Afternoon",
                "Night", "Morning", "Evening", "Afternoon",
            ],
            "Vehicle_Type": [
                "Premium", "Economy", "Premium", "Economy",
                "Premium", "Economy", "Premium", "Economy",
            ],
            "Expected_Ride_Duration": [90, 60, 45, 100, 80, 30, 150, 70],
            "Historical_Cost_of_Ride": [300.0, 250.0, 220.0, 400.0, 310.0, 150.0, 480.0, 260.0],
        }
    )

    data_with_ratio = engineer_demand_supply_ratio(cleaned_data, parameters)
    assert "demand_supply_ratio" in data_with_ratio.columns

    encoder = fit_categorical_encoder(data_with_ratio, parameters)
    encoded_data = encode_categorical_features(data_with_ratio, encoder, parameters)

    for col in parameters["categorical_columns"]:
        assert col not in encoded_data.columns

    feature_table = assemble_feature_table(encoded_data, parameters)

    assert feature_table.columns[-1] == parameters["target_column"]
    assert feature_table.isnull().sum().sum() == 0
    assert len(feature_table) == len(cleaned_data)

    print("Full feature_engineering chain executed successfully end-to-end.")
    print(feature_table.head())


if __name__ == "__main__":
    test_full_feature_engineering_chain_runs_end_to_end()
