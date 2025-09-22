"""
Main FastAPI application for the Alerting Server.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dateutil.parser import isoparse
from models import AlertPayload
from database import save_alert, get_alerts, get_alert_by_id, get_total_alerts_count

app = FastAPI(
    title="IoT Anomaly Alerting API",
    description="Receives, validates, and stores anomaly alerts from sensors.",
    version="1.0.0"
)

# --- Templating and Static Files Setup ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Alerting API is running."}

@app.get("/dashboard", response_class=HTMLResponse, tags=["Monitoring"])
def view_dashboard(request: Request, page: int = 1, per_page: int = 12):
    """
    Renders the monitoring dashboard with pagination support.
    """
    # Calculate offset based on page number
    offset = (page - 1) * per_page

    # Get paginated alerts and total count
    alerts_data = get_alerts(limit=per_page, offset=offset)
    total_alerts = get_total_alerts_count()

    # Calculate pagination info
    total_pages = (total_alerts + per_page - 1) // per_page  # Ceiling division
    has_prev = page > 1
    has_next = page < total_pages

    pagination_info = {
        "current_page": page,
        "per_page": per_page,
        "total_alerts": total_alerts,
        "total_pages": total_pages,
        "has_prev": has_prev,
        "has_next": has_next,
        "prev_page": page - 1 if has_prev else None,
        "next_page": page + 1 if has_next else None,
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "alerts": alerts_data,
            "pagination": pagination_info
        }
    )

@app.get("/dashboard/alert/{alert_id}", response_class=HTMLResponse, tags=["Monitoring"])
def view_alert_detail(request: Request, alert_id: str):
    """
    Renders the detailed view for a specific alert, including visualization.
    """
    alert = get_alert_by_id(alert_id)

    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    return templates.TemplateResponse(
        "alert_detail.html",
        {"request": request, "alert": alert}
    )

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
