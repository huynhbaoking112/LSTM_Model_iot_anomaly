import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

class LSTMAutoencoder:
    def __init__(self, sequence_length=24, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.threshold = None
        
    def build_model(self):
        """Xây dựng LSTM Autoencoder"""
        model = Sequential([
            # Encoder
            LSTM(64, activation='relu', input_shape=(self.sequence_length, self.n_features), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            RepeatVector(self.sequence_length),
            
            # Decoder
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(self.n_features))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model
    
    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """Huấn luyện model"""
        if self.model is None:
            self.build_model()
            
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/best_lstm_model.h5', monitor='val_loss', 
                          save_best_only=True)
        ]
        
        history = self.model.fit(
            train_data, train_data,  # Autoencoder: input = target
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, val_data),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, data):
        """Dự đoán và tính reconstruction error"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Get reconstructions
        reconstructions = self.model.predict(data)
        
        # Calculate MSE for each sequence
        mse = np.mean(np.power(data - reconstructions, 2), axis=(1, 2))
        
        return reconstructions, mse
    
    def set_threshold(self, val_data, percentile=95):
        """Đặt threshold dựa trên validation data"""
        _, val_errors = self.predict(val_data)
        
        # Set threshold at percentile of validation errors
        self.threshold = np.percentile(val_errors, percentile)
        print(f"Anomaly threshold set to: {self.threshold}")
        
        return self.threshold
    
    def detect_anomalies(self, data):
        """Phát hiện anomalies"""
        _, errors = self.predict(data)
        
        # Anomalies are points above threshold
        anomalies = errors > self.threshold
        
        return anomalies, errors
    
    def save_model(self, filepath):
        """Lưu model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")

def plot_training_history(history):
    """Vẽ biểu đồ training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def plot_anomalies(data, anomalies, errors, save_path='models/anomalies_plot.png'):
    """Vẽ biểu đồ anomalies"""
    plt.figure(figsize=(15, 6))
    
    # Plot original data
    plt.subplot(1, 2, 1)
    plt.plot(data.flatten())
    plt.title('Original Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Value')
    
    # Plot reconstruction errors
    plt.subplot(1, 2, 2)
    plt.plot(errors, 'r-', label='Reconstruction Error')
    plt.axhline(y=errors[anomalies].min() if np.any(anomalies) else 0, 
               color='k', linestyle='--', label='Anomaly Threshold')
    plt.scatter(np.where(anomalies)[0], errors[anomalies], 
               color='red', marker='x', s=50, label='Anomalies')
    plt.title('Reconstruction Errors and Detected Anomalies')
    plt.xlabel('Sequences')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
