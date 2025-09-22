"""
Module for running inference on a custom CSV file using the trained model.
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH, ANOMALY_THRESHOLD_PERCENTILE, 
    TRAIN_SPLIT, VAL_SPLIT, ALERTING_ENABLED, ALERTING_API_ENDPOINT, SENSOR_ID
)
from src.preprocessor import DataPreprocessor

def send_alert_to_server(payload):
    """Sends the alert payload to the configured API endpoint."""
    if not ALERTING_ENABLED:
        print("   -> Alerting is disabled. Skipping notification.")
        return

    try:
        print(f"   -> 🚨 Sending anomaly alert to {ALERTING_API_ENDPOINT}...")
        response = requests.post(ALERTING_API_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        print("      -> ✅ Alert successfully sent.")
    except requests.exceptions.RequestException as e:
        print(f"      -> ❌ Failed to send alert: {e}")

class Predictor:
    """
    Handles loading the model and running predictions on new data.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None

    def load_artifacts(self):
        """Loads the trained model, scaler, and calculates the threshold."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print(f"❌ Error: Model or scaler not found. Please run the training pipeline first.")
            return False
        
        print("🔄 Loading model and scaler...")
        # Apply the same fix as in lstm_model.py
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'mae': tf.keras.losses.MeanAbsoluteError()}
        )
        self.scaler = joblib.load(SCALER_PATH)

        print("   -> Recalculating anomaly threshold from original validation set...")
        preprocessor = DataPreprocessor()
        _, val_X, _, _ = preprocessor.run()
        if val_X is None:
            print("❌ Could not get validation data to set threshold.")
            return False
        
        val_reconstructions = self.model.predict(val_X, verbose=0)
        val_mae = np.mean(np.abs(val_reconstructions - val_X), axis=1)
        self.threshold = np.percentile(val_mae, ANOMALY_THRESHOLD_PERCENTILE)
        print(f"   -> Anomaly threshold consistently set to {self.threshold:.4f}")
        return True

    def predict_anomalies(self, csv_path):
        """
        Loads a custom CSV, preprocesses it, and predicts anomalies.
        """
        if not os.path.exists(csv_path):
            print(f"❌ Error: Input file not found at '{csv_path}'.")
            return None, None

        print(f"\n🔬 Processing custom file: '{csv_path}'")
        # 1. Load and preprocess custom data
        # Use header=0 to automatically use the first row as headers, or no header if file has none.
        # The names parameter will be used if there's no header.
        try:
            df = pd.read_csv(
                csv_path, 
                header=0, 
                names=['timestamp', 'value'], 
                parse_dates=['timestamp'], 
                index_col='timestamp'
            )
            # If the header was 'timestamp', 'value', pandas might create a MultiIndex. Let's fix that.
            if isinstance(df.index, pd.MultiIndex):
                 df = pd.read_csv(
                    csv_path, 
                    header=None, 
                    names=['timestamp', 'value'], 
                    parse_dates=['timestamp'], 
                    index_col='timestamp',
                    skiprows=1
                )
        except Exception:
             # Fallback for files with no header as originally instructed
             df = pd.read_csv(
                csv_path, 
                header=None, 
                names=['timestamp', 'value'], 
                parse_dates=['timestamp'], 
                index_col='timestamp'
            )
        
        # --- NEW: Ensure timestamps are UTC-aware ---
        # This standardizes the timestamp format before sending and avoids parsing errors.
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        # --- END NEW ---

        # Clean up any non-numeric values that might have slipped through
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True)

        if len(df) < SEQUENCE_LENGTH:
            print(f"❌ Error: Input data must have at least {SEQUENCE_LENGTH} records to create a sequence.")
            return None, None

        scaled_values = self.scaler.transform(df[['value']])

        sequences = []
        for i in range(len(scaled_values) - SEQUENCE_LENGTH + 1):
            sequences.append(scaled_values[i:(i + SEQUENCE_LENGTH)])
        sequences = np.array(sequences)
        
        reconstructions = self.model.predict(sequences, verbose=0)
        mae = np.mean(np.abs(reconstructions - sequences), axis=1).flatten()
        
        anomalies_mask = mae > self.threshold
        
        # --- NEW: Send alerts for each detected anomaly ---
        print(f"   -> Found {np.sum(anomalies_mask)} anomalous sequences.")
        if np.sum(anomalies_mask) > 0:
            # Reshape for inverse_transform: from (n_sequences, seq_len, 1) to (n_sequences * seq_len, 1)
            # Then reshape back to (n_sequences, seq_len)
            original_sequences_inv = self.scaler.inverse_transform(
                sequences.reshape(-1, 1)
            ).reshape(sequences.shape[0], SEQUENCE_LENGTH)
            
            reconstructed_sequences_inv = self.scaler.inverse_transform(
                reconstructions.reshape(-1, 1)
            ).reshape(reconstructions.shape[0], SEQUENCE_LENGTH)

            for i, is_anomaly in enumerate(anomalies_mask):
                if is_anomaly:
                    sequence_timestamps = df.index[i : i + SEQUENCE_LENGTH]
                    
                    payload = {
                        "sensor_id": SENSOR_ID,
                        "alert_timestamp": sequence_timestamps[-1].isoformat(), # No more manual "+ Z"
                        "anomaly_score": float(mae[i]),
                        "threshold": float(self.threshold),
                        "sequence_data": {
                            "timestamps": [ts.isoformat() for ts in sequence_timestamps], # No more manual "+ Z"
                            "original_values": original_sequences_inv[i].flatten().tolist(),
                            "reconstructed_values": reconstructed_sequences_inv[i].flatten().tolist(),
                        }
                    }
                    send_alert_to_server(payload)
        # --- END NEW ---
        
        # --- NEW: Create reconstructed series for plotting ---
        reconstructions_inv = self.scaler.inverse_transform(reconstructions.reshape(-1, 1)).reshape(sequences.shape[0], SEQUENCE_LENGTH)
        
        results_df = df.copy()
        results_df = results_df.rename(columns={'value': 'NO2_original'})
        results_df['NO2_reconstructed'] = np.nan

        # Populate reconstructed values, overwriting for overlaps. 
        # The last sequence's reconstruction for a point is what gets shown.
        for i in range(len(reconstructions_inv)):
            results_df.iloc[i:i+SEQUENCE_LENGTH, results_df.columns.get_loc('NO2_reconstructed')] = reconstructions_inv[i].flatten()
        # --- END NEW ---

        anomaly_points = np.zeros(len(df), dtype=bool)
        for i, is_anomaly in enumerate(anomalies_mask):
            if is_anomaly:
                anomaly_points[i : i + SEQUENCE_LENGTH] = True
        results_df['is_anomaly'] = anomaly_points

        error_df = pd.DataFrame(index=df.index[:len(mae)])
        error_df['mae'] = mae
        error_df['threshold'] = self.threshold
        error_df['is_anomaly'] = anomalies_mask
        
        return results_df, error_df

    def plot_predictions(self, results_df, error_df, input_filename):
        """Plots the prediction results in the detailed format."""
        print("📈 Generating detailed prediction plots...")
        
        anomalies_df = results_df[results_df['is_anomaly']]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(f'Detailed Anomaly Prediction for "{os.path.basename(input_filename)}"', fontsize=18)

        # --- Plot 1: Original vs. Reconstructed ---
        ax1.plot(results_df.index, results_df['NO2_original'], label='Input NO2 Levels', color='blue', zorder=1)
        ax1.plot(results_df.index, results_df['NO2_reconstructed'], label='Reconstructed by Model', color='orange', linestyle='--', zorder=2)
        ax1.scatter(anomalies_df.index, anomalies_df['NO2_original'], color='red', s=50, label='Detected Anomaly Point', zorder=5)
        ax1.set_title('Input Data vs. Model Reconstruction')
        ax1.set_ylabel('NO2 (μg/m³)')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Plot 2: Reconstruction Error ---
        ax2.plot(error_df.index, error_df['mae'], label='Reconstruction Error (MAE)', color='green')
        ax2.axhline(self.threshold, color='red', linestyle='--', label=f'Anomaly Threshold ({self.threshold:.4f})')
        anomaly_errors = error_df[error_df['is_anomaly']]
        ax2.scatter(anomaly_errors.index, anomaly_errors['mae'], color='red', s=50)
        ax2.set_title('Reconstruction Error for Each Input Sequence')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Mean Absolute Error (MAE)')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        clean_filename = "".join([c for c in os.path.basename(input_filename) if c.isalpha() or c.isdigit()]).rstrip()
        output_path = f'reports/figures/prediction_{clean_filename}.png'
        plt.savefig(output_path)
        print(f"   -> Prediction plot saved to '{output_path}'")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run anomaly prediction on a custom CSV file.")
    parser.add_argument(
        'input_file', 
        type=str,
        help="Path to the custom CSV file. Format: 2 columns ('timestamp', 'value'), no header."
    )
    args = parser.parse_args()

    predictor = Predictor()
    if predictor.load_artifacts():
        results, errors = predictor.predict_anomalies(args.input_file)
        if results is not None:
            predictor.plot_predictions(results, errors, args.input_file)
            print("\n✅ Prediction complete.")

if __name__ == '__main__':
    main()
