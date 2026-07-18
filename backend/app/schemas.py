"""Request/response schemas.

`RideRequest` mirrors `model/conf/base/parameters_inference.yml`'s
`expected_schema` exactly (minus the target column) and enforces the same
value bounds as `data_processing`'s `value_ranges` — so an invalid
request is rejected by FastAPI's validation layer before it ever reaches
the model, with a clear 422 response instead of a confusing downstream
failure.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RideRequest(BaseModel):
    Number_of_Riders: int = Field(ge=0, le=500)
    Number_of_Drivers: int = Field(gt=0, le=500)  # > 0: a ratio by zero is undefined
    Location_Category: Literal["Urban", "Suburban", "Rural"]
    Customer_Loyalty_Status: Literal["Silver", "Regular", "Gold"]
    Number_of_Past_Rides: int = Field(ge=0, le=1000)
    Average_Ratings: float = Field(ge=1.0, le=5.0)
    Time_of_Booking: Literal["Morning", "Afternoon", "Evening", "Night"]
    Vehicle_Type: Literal["Economy", "Premium"]
    Expected_Ride_Duration: int = Field(gt=0, le=600)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Number_of_Riders": 90,
                "Number_of_Drivers": 45,
                "Location_Category": "Urban",
                "Customer_Loyalty_Status": "Silver",
                "Number_of_Past_Rides": 10,
                "Average_Ratings": 4.5,
                "Time_of_Booking": "Night",
                "Vehicle_Type": "Premium",
                "Expected_Ride_Duration": 90,
            }
        }
    }


class PredictionResponse(BaseModel):
    request_id: str
    predicted_fare: float
    is_out_of_bounds: bool
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None
    production_alias: str


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str
