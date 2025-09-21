# FIWARE API Configuration
FIWARE_BASE_URL = "https://api.smartsantander.eu"
FIWARE_ENTITY_TYPE = "AirQualityObserved"

# Data Collection
COLLECTION_INTERVAL_MINUTES = 15
DATA_LIMIT = 1000

# Model Configuration
SEQUENCE_LENGTH = 24  # 24 hours for LSTM
PREDICTION_HORIZON = 1
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Thresholds for Anomaly Detection
ANOMALY_THRESHOLD = 0.1  # Threshold for reconstruction error

# File Paths
RAW_DATA_PATH = "data/raw/air_quality_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/lstm_anomaly_detector.h5"
