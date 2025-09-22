"""
Handles all database operations, including connecting to MongoDB
and saving alert documents.
"""
from pymongo import MongoClient, DESCENDING
from bson.objectid import ObjectId
from pymongo.errors import ConnectionFailure, OperationFailure
from config import MONGO_CONNECTION_STRING

DB_NAME = "iot_anomaly_alerts"
COLLECTION_NAME = "alerts"

# --- Database Client ---
# Initialize the client once to be reused across the application
try:
    if not MONGO_CONNECTION_STRING:
        raise ValueError("MONGO_CONNECTION_STRING is not set in the config file.")
    
    client = MongoClient(MONGO_CONNECTION_STRING)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("✅ Successfully connected to MongoDB.")
    db = client[DB_NAME]
    alerts_collection = db[COLLECTION_NAME]

except (ConnectionFailure, ValueError) as e:
    print(f"❌ Could not connect to MongoDB: {e}")
    client = None
    alerts_collection = None

def save_alert(alert_data: dict) -> (bool, str):
    """
    Saves a single alert document to the 'alerts' collection.

    Args:
        alert_data (dict): A dictionary representing the alert payload.

    Returns:
        tuple[bool, str]: A tuple containing a success flag and a message
                          (either the document ID or an error message).
    """
    if alerts_collection is None:
        error_message = "Database collection is not available."
        print(f"   -> ❌ {error_message}")
        return False, error_message

    try:
        result = alerts_collection.insert_one(alert_data)
        inserted_id = str(result.inserted_id)
        print(f"   -> ✅ Alert saved to DB with ID: {inserted_id}")
        return True, inserted_id
    except OperationFailure as e:
        error_message = f"Failed to save alert to DB: {e}"
        print(f"   -> ❌ {error_message}")
        return False, error_message

def get_alerts(limit: int = 50, offset: int = 0) -> list:
    """
    Retrieves alerts from the database with pagination support.

    Args:
        limit (int): The maximum number of alerts to retrieve.
        offset (int): The number of alerts to skip for pagination.

    Returns:
        list: A list of alert documents, or an empty list if none are found or on error.
    """
    if alerts_collection is None:
        print("   -> ❌ Database collection is not available.")
        return []

    try:
        # Sort by 'alert_timestamp' in descending order to get the newest first
        alerts = alerts_collection.find().sort("alert_timestamp", DESCENDING).skip(offset).limit(limit)
        return list(alerts)
    except OperationFailure as e:
        print(f"   -> ❌ Failed to fetch alerts from DB: {e}")
        return []

def get_total_alerts_count() -> int:
    """
    Gets the total count of alerts in the database.

    Returns:
        int: The total number of alerts.
    """
    if alerts_collection is None:
        return 0

    try:
        return alerts_collection.count_documents({})
    except OperationFailure as e:
        print(f"   -> ❌ Failed to count alerts: {e}")
        return 0

def get_alert_by_id(alert_id: str) -> dict:
    """
    Retrieves a single alert by its ID.

    Args:
        alert_id (str): The MongoDB ObjectId of the alert.

    Returns:
        dict: The alert document, or None if not found or on error.
    """
    if alerts_collection is None:
        print("   -> ❌ Database collection is not available.")
        return None

    try:
        # Convert string ID to ObjectId
        obj_id = ObjectId(alert_id)
        alert = alerts_collection.find_one({"_id": obj_id})
        return alert
    except Exception as e:
        print(f"   -> ❌ Failed to fetch alert {alert_id}: {e}")
        return None
