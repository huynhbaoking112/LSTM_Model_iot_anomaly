# Configuration file for NO2 Anomaly Detection System

# FIWARE API Configuration
FIWARE_BASE_URL = "https://api.smartsantander.eu"
FIWARE_ENTITY_TYPE = "AirQualityObserved"

# NO2-specific Configuration
TARGET_ATTRIBUTE = "NO2"  # Focus on NO2 gas detection
NO2_NORMAL_RANGE = (0, 200)  # Normal NO2 range in μg/m³
NO2_OUTLIER_THRESHOLD = 400  # Values above this are likely errors

# Data Collection
COLLECTION_INTERVAL_MINUTES = 15
DATA_LIMIT = 1000
HISTORICAL_DAYS = 30  # Collect 30 days of historical data

# Data Preprocessing
SEQUENCE_LENGTH = 24      # Use 24 hours of data to predict the next state
TRAIN_SPLIT = 0.7         # 70% of data for training
VAL_SPLIT = 0.15          # 15% for validation, 15% for testing

# Model Configuration for NO2 Time Series
PREDICTION_HORIZON = 1
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# NO2-specific LSTM Parameters
LSTM_UNITS_ENCODER = 64
LSTM_UNITS_DECODER = 32
DROPOUT_RATE = 0.2

# Thresholds for NO2 Anomaly Detection
ANOMALY_THRESHOLD_PERCENTILE = 95  # Use 95th percentile of validation errors
NO2_SPIKE_MULTIPLIER = 3.0  # Detect spikes > 3x normal levels

# NO2 Temporal Pattern Configuration
RUSH_HOURS = [(7, 9), (17, 19)]  # Morning and evening rush hours
WEEKEND_PATTERN_DIFF = True  # Account for weekend vs weekday differences

# File Paths
RAW_DATA_PATH = "data/raw/madrid_2018.csv"
PROCESSED_DATA_PATH = "data/processed/madrid_station_28079004_no2.csv"
MODEL_PATH = "models/no2_lstm_anomaly_detector.h5"
SCALER_PATH = "models/no2_scaler.joblib"

# Visualization Settings
PLOT_DPI = 300
FIGURE_SIZE = (15, 8)
SAVE_PLOTS = True

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API Rate Limiting
API_DELAY_SECONDS = 1  # Delay between API calls to avoid rate limiting
MAX_RETRIES = 3  # Maximum number of retries for failed API calls
