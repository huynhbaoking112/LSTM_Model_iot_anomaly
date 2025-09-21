"""
Module for building, training, and using the LSTM Autoencoder model.
"""
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (MODEL_PATH, SEQUENCE_LENGTH, LSTM_UNITS_ENCODER, 
                    LSTM_UNITS_DECODER, DROPOUT_RATE, BATCH_SIZE, EPOCHS, 
                    ANOMALY_THRESHOLD_PERCENTILE)
from src.preprocessor import DataPreprocessor

class LSTMAutoencoder:
    """
    Handles the creation, training, and application of the LSTM Autoencoder model.
    """
    def __init__(self):
        self.model = None
        self.history = None
        self.threshold = None

    def build_model(self):
        """Builds the LSTM Autoencoder architecture."""
        print("🏗️ Building LSTM Autoencoder model...")
        
        model = Sequential()
        # --- Encoder ---
        model.add(LSTM(
            units=LSTM_UNITS_ENCODER[0],
            activation='relu',
            input_shape=(SEQUENCE_LENGTH, 1),
            return_sequences=True
        ))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(LSTM(
            units=LSTM_UNITS_ENCODER[1],
            activation='relu',
            return_sequences=False
        ))
        model.add(Dropout(rate=DROPOUT_RATE))

        # --- Bottleneck ---
        model.add(RepeatVector(SEQUENCE_LENGTH))

        # --- Decoder ---
        model.add(LSTM(
            units=LSTM_UNITS_DECODER[0],
            activation='relu',
            return_sequences=True
        ))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(LSTM(
            units=LSTM_UNITS_DECODER[1],
            activation='relu',
            return_sequences=True
        ))
        model.add(Dropout(rate=DROPOUT_RATE))

        # --- Output Layer ---
        model.add(TimeDistributed(Dense(units=1)))
        
        model.compile(optimizer='adam', loss='mae') # MAE is often more robust to outliers
        model.summary()
        self.model = model

    def train(self, train_data, val_data):
        """Trains the model and saves the best version."""
        if self.model is None:
            self.build_model()

        print("🧠 Training model...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True),
            ModelCheckpoint(
                filepath=MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]

        self.history = self.model.fit(
            train_data, train_data, # Autoencoder learns to reconstruct its input
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(val_data, val_data),
            callbacks=callbacks,
            shuffle=False # Important for time series
        )
        print(f"✅ Model training complete. Best model saved to '{MODEL_PATH}'")

    def plot_training_history(self):
        """Plots the training and validation loss."""
        if self.history is None:
            print("⚠️ No training history found. Please train the model first.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.ylabel('Loss (MAE)')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Save the plot
        os.makedirs('reports/figures', exist_ok=True)
        plot_path = 'reports/figures/training_history.png'
        plt.savefig(plot_path)
        print(f"   -> Training history plot saved to '{plot_path}'")
        plt.close()

    def set_anomaly_threshold(self, val_data):
        """
        Sets the anomaly detection threshold based on reconstruction errors
        on the validation dataset.
        """
        print("📊 Calculating anomaly threshold...")
        # Get reconstruction errors for validation data
        val_reconstructions = self.model.predict(val_data)
        val_mae = np.mean(np.abs(val_reconstructions - val_data), axis=1)

        # Set threshold at a high percentile
        self.threshold = np.percentile(val_mae, ANOMALY_THRESHOLD_PERCENTILE)
        print(f"   -> Anomaly threshold (MAE > {self.threshold:.4f}) set based on {ANOMALY_THRESHOLD_PERCENTILE}th percentile of validation errors.")
        
        # Plot distribution of reconstruction errors
        plt.figure(figsize=(10, 5))
        sns.histplot(val_mae, bins=50, kde=True)
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold = {self.threshold:.4f}')
        plt.title('Distribution of Reconstruction Errors on Validation Set')
        plt.xlabel('Mean Absolute Error (MAE)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plot_path = 'reports/figures/reconstruction_error_distribution.png'
        plt.savefig(plot_path)
        print(f"   -> Error distribution plot saved to '{plot_path}'")
        plt.close()

    def load_model(self):
        """Loads a pre-trained model from disk."""
        if os.path.exists(MODEL_PATH):
            print(f"🔄 Loading saved model from '{MODEL_PATH}'...")
            # Provide the custom object dictionary to help Keras find the 'mae' function
            # This is necessary for newer versions of TensorFlow/Keras
            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'mae': tf.keras.losses.MeanAbsoluteError()}
            )
        else:
            print(f"⚠️ No saved model found at '{MODEL_PATH}'.")

def run_training_pipeline():
    """
    Orchestrates the full pipeline: Preprocessing -> Training -> Threshold Setting.
    """
    print("====== Starting Full Training Pipeline ======")
    
    # 1. Preprocess the data
    preprocessor = DataPreprocessor()
    train_X, val_X, test_X, _ = preprocessor.run()
    
    if train_X is None:
        print("❌ Halting pipeline due to preprocessing failure.")
        return

    # 2. Build and train the model
    autoencoder = LSTMAutoencoder()
    autoencoder.build_model()
    autoencoder.train(train_X, val_X)
    
    # 3. Plot training history
    autoencoder.plot_training_history()
    
    # 4. Set anomaly threshold
    # The model is automatically loaded from the checkpoint of the best validation loss
    autoencoder.load_model() 
    autoencoder.set_anomaly_threshold(val_X)
    
    print("\n====== Full Training Pipeline Completed Successfully! ======")
    print("Next step: Use the trained model and threshold to evaluate on the test set (Giai đoạn 5).")

if __name__ == '__main__':
    run_training_pipeline()
