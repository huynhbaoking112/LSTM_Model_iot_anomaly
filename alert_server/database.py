"""
Handles all database operations, including connecting to MongoDB
and saving alert documents.
"""
from pymongo import MongoClient
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
