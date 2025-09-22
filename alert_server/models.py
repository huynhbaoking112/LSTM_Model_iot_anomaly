"""
Pydantic models for API data validation.
"""
from pydantic import BaseModel
from typing import List
# We will accept string dates from the client and convert them in the API logic.
# from datetime import datetime

class SequenceData(BaseModel):
    """
    Represents the detailed time-series data within an anomalous sequence.
    """
    timestamps: List[str]  # Expect ISO 8601 formatted strings
    original_values: List[float]
    reconstructed_values: List[float]

class AlertPayload(BaseModel):
    """
    Defines the structure for an incoming alert from the predictor.
    FastAPI will use this model to validate the request body.
    """
    sensor_id: str
    alert_timestamp: str  # Expect an ISO 8601 formatted string
    anomaly_score: float
    threshold: float
    sequence_data: SequenceData
