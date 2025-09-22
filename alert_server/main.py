"""
Main FastAPI application for the Alerting Server.
"""
from fastapi import FastAPI, HTTPException
from dateutil.parser import isoparse
from models import AlertPayload
from database import save_alert

app = FastAPI(
    title="IoT Anomaly Alerting API",
    description="Receives, validates, and stores anomaly alerts from sensors.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Alerting API is running."}

@app.post("/api/alerts", tags=["Alerts"], status_code=201)
def create_alert(payload: AlertPayload):
    """
    Receives an anomaly alert, validates its structure,
    and saves it to the database.
    """
    print(f"Received alert for sensor: {payload.sensor_id}")
    
    # Pydantic models have a .model_dump() method to convert them to dictionaries
    alert_dict = payload.model_dump()
    
    # --- Data Conversion ---
    # Manually convert string timestamps to datetime objects before DB insertion.
    # This is crucial for MongoDB to store them as a native BSON date type.
    try:
        alert_dict['alert_timestamp'] = isoparse(alert_dict['alert_timestamp'])
        alert_dict['sequence_data']['timestamps'] = [
            isoparse(ts) for ts in alert_dict['sequence_data']['timestamps']
        ]
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp format: {e}"
        )
    # --- End Conversion ---

    success, result = save_alert(alert_dict)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save alert to database: {result}"
        )
        
    return {
        "status": "success",
        "message": "Alert received and stored successfully.",
        "alert_id": result
    }
