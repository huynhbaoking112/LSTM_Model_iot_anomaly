import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt
from config import *

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, filepath):
        """Load dữ liệu từ CSV"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df):
        """Làm sạch dữ liệu"""
        # Remove missing values
        df = df.dropna()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        print(f"After cleaning: {len(df)} records")
        return df
    
    def preprocess_for_lstm(self, df, target_attribute='NO2'):
        """Tiền xử lý dữ liệu cho LSTM"""
        # Filter by target attribute
        df_target = df[df['attribute'] == target_attribute].copy()
        
        if df_target.empty:
            raise ValueError(f"No data found for attribute: {target_attribute}")
        
        # Pivot to have timestamps as rows and attributes as columns
        # For now, we'll work with single attribute, later extend to multivariate
        
        # Normalize values
        values = df_target['value'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences for LSTM
        sequences = self._create_sequences(scaled_values, SEQUENCE_LENGTH)
        
        return sequences, self.scaler
    
    def _create_sequences(self, data, seq_length):
        """Tạo sequences cho LSTM"""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:(i + seq_length)])
        return np.array(sequences)
    
    def split_data(self, sequences, train_ratio=0.7, val_ratio=0.15):
        """Chia dữ liệu thành train, validation, test"""
        n = len(sequences)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = sequences[:train_size]
        val_data = sequences[train_size:train_size + val_size]
        test_data = sequences[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def prepare_data_for_training(self, filepath, target_attribute='NO2'):
        """Pipeline hoàn chỉnh để chuẩn bị dữ liệu"""
        df = self.load_data(filepath)
        df = self.clean_data(df)
        sequences, scaler = self.preprocess_for_lstm(df, target_attribute)
        train_data, val_data, test_data = self.split_data(sequences)
        
        print(f"Train shape: {train_data.shape}")
        print(f"Validation shape: {val_data.shape}")
        print(f"Test shape: {test_data.shape}")
        
        return train_data, val_data, test_data, scaler
