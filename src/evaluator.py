"""
Module for evaluating the trained LSTM Autoencoder model on the test dataset.
"""
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, SCALER_PATH
from src.preprocessor import DataPreprocessor
from src.lstm_model import LSTMAutoencoder

class ModelEvaluator:
    """
    Handles loading the trained model and evaluating its performance on the test set.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None # This will be loaded from the trained model logic

    def load_artifacts(self):
        """Loads the trained model and scaler."""
        # Load model
        if os.path.exists(MODEL_PATH):
            print(f"🔄 Loading trained model from '{MODEL_PATH}'...")
            # Apply the fix for the 'mae' deserialization error
            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'mae': tf.keras.losses.MeanAbsoluteError()}
            )
        else:
            print(f"❌ Error: Model file not found at '{MODEL_PATH}'. Please run the training pipeline first.")
            return False

        # Load scaler
        if os.path.exists(SCALER_PATH):
            print(f"🔄 Loading scaler from '{SCALER_PATH}'...")
            self.scaler = joblib.load(SCALER_PATH)
        else:
            print(f"❌ Error: Scaler file not found at '{SCALER_PATH}'. Please run the preprocessing pipeline first.")
            return False
            
        return True

    def evaluate(self, test_data, val_data):
        """
        Evaluates the model on the test data.
        Args:
            test_data: The preprocessed test data sequences.
            val_data: The preprocessed validation data to determine the threshold.
        Returns:
            A tuple of (results_df, error_df) for plotting.
        """
        print("\n🔬 Evaluating model performance on the test set...")
        print("   -> Calculating threshold from validation data...")
        val_reconstructions = self.model.predict(val_data, verbose=0)
        val_mae = np.mean(np.abs(val_reconstructions - val_data), axis=1)
        from config import ANOMALY_THRESHOLD_PERCENTILE
        self.threshold = np.percentile(val_mae, ANOMALY_THRESHOLD_PERCENTILE)
        print(f"   -> Anomaly threshold set to {self.threshold:.4f}")

        print("   -> Calculating reconstruction error on test data...")
        test_reconstructions = self.model.predict(test_data, verbose=0)
        test_mae = np.mean(np.abs(test_reconstructions - test_data), axis=1).flatten()
        anomalies_mask = test_mae > self.threshold
        print(f"   -> Found {np.sum(anomalies_mask)} anomalous sequences out of {len(test_data)}.")
        
        # --- REVISED LOGIC FOR ROBUST INDEXING ---
        from config import PROCESSED_DATA_PATH, TRAIN_SPLIT, VAL_SPLIT, SEQUENCE_LENGTH
        
        # 1. Get the full, resampled and imputed dataframe from the preprocessor
        # This ensures we have the exact same data that was used to create the sequences
        preprocessor_for_index = DataPreprocessor()
        full_df = preprocessor_for_index._load_data(PROCESSED_DATA_PATH)
        resampled_df = preprocessor_for_index._resample_and_impute(full_df)
        
        # 2. Determine the correct split points on the resampled data
        n_sequences = len(resampled_df) - SEQUENCE_LENGTH
        train_end_idx = int(n_sequences * TRAIN_SPLIT)
        val_end_idx = train_end_idx + int(n_sequences * VAL_SPLIT)
        
        # 3. Get the correct time index for the test data points
        # The test data starts after the validation sequences end
        test_start_point = val_end_idx
        test_time_index = resampled_df.index[test_start_point : test_start_point + len(test_data.flatten())]
        
        original_values = self.scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()
        reconstructed_values = self.scaler.inverse_transform(test_reconstructions.reshape(-1, 1)).flatten()
        
        # Ensure lengths match before creating DataFrame
        min_len = min(len(test_time_index), len(original_values))
        results_df = pd.DataFrame(index=test_time_index[:min_len])
        results_df['NO2_original'] = original_values[:min_len]
        results_df['NO2_reconstructed'] = reconstructed_values[:min_len]
        
        # Map sequence anomalies to points
        anomaly_points = np.zeros(len(results_df), dtype=bool)
        for i, is_anomaly in enumerate(anomalies_mask):
            if is_anomaly:
                anomaly_points[i : i + SEQUENCE_LENGTH] = True
        results_df['is_anomaly'] = anomaly_points[:len(results_df)]
        
        # 4. Get the correct time index for the error dataframe (start of each test sequence)
        error_df_index = resampled_df.index[val_end_idx : val_end_idx + len(test_mae)]
        error_df = pd.DataFrame(index=error_df_index)
        error_df['mae'] = test_mae
        error_df['threshold'] = self.threshold
        error_df['is_anomaly'] = anomalies_mask
        
        return results_df, error_df

    def plot_detailed_results(self, results_df, error_df):
        """
        Visualizes the evaluation results in a detailed, two-panel plot.
        """
        print("📈 Generating detailed evaluation plots...")
        
        anomalies_df = results_df[results_df['is_anomaly']]
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=(18, 12), 
            sharex=True, 
            gridspec_kw={'height_ratios': [2, 1]}
        )
        fig.suptitle('Detailed Anomaly Detection Evaluation on Test Set', fontsize=18)

        # --- Plot 1: Original vs. Reconstructed ---
        ax1.plot(results_df.index, results_df['NO2_original'], label='Original NO2 Levels', color='blue', alpha=0.9, zorder=1)
        ax1.plot(results_df.index, results_df['NO2_reconstructed'], label='Reconstructed by Model', color='orange', linestyle='--', zorder=2)
        ax1.scatter(anomalies_df.index, anomalies_df['NO2_original'], color='red', s=40, label='Detected Anomaly Point', zorder=3)
        
        ax1.set_title('Original vs. Reconstructed NO2 Levels')
        ax1.set_ylabel('NO2 (μg/m³)')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Plot 2: Reconstruction Error ---
        ax2.plot(error_df.index, error_df['mae'], label='Reconstruction Error (MAE)', color='green')
        ax2.axhline(self.threshold, color='red', linestyle='--', label=f'Anomaly Threshold ({self.threshold:.4f})')
        
        anomaly_errors = error_df[error_df['is_anomaly']]
        ax2.scatter(anomaly_errors.index, anomaly_errors['mae'], color='red', s=40, label='Anomalous Sequence')
        
        ax2.set_title('Reconstruction Error for Each Sequence')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Mean Absolute Error (MAE)')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.set_yscale('log') # Use log scale to better visualize variations in error

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save the plot
        plot_path = 'reports/figures/evaluation_detailed_results.png'
        plt.savefig(plot_path)
        print(f"   -> Detailed evaluation plot saved to '{plot_path}'")
        plt.close()
        
        # Print summary
        print("\n--- Evaluation Summary ---")
        print(f"Total test sequences: {len(error_df)}")
        print(f"Total anomalous sequences detected: {np.sum(error_df['is_anomaly'])}")
        if len(error_df) > 0:
            anomaly_rate = np.mean(error_df['is_anomaly']) * 100
            print(f"Anomaly Rate (by sequence): {anomaly_rate:.2f}%")

def run_evaluation_pipeline():
    """
    Orchestrates the full evaluation pipeline.
    """
    print("\n====== Starting Full Evaluation Pipeline ======")
    
    # 1. Load artifacts
    evaluator = ModelEvaluator()
    if not evaluator.load_artifacts():
        return
        
    # 2. Get the preprocessed data
    preprocessor = DataPreprocessor()
    _, val_X, test_X, _ = preprocessor.run()
    
    if test_X is None:
        print("❌ Halting pipeline due to preprocessing failure.")
        return
        
    # 3. Perform evaluation
    results_df, error_df = evaluator.evaluate(test_X, val_X)
    
    # 4. Plot results
    if not results_df.empty:
        evaluator.plot_detailed_results(results_df, error_df)

    print("\n====== Full Evaluation Pipeline Completed Successfully! ======")
    print("Check 'reports/figures/evaluation_detailed_results.png' for the new, detailed output.")

if __name__ == '__main__':
    run_evaluation_pipeline()
