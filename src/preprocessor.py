"""
Module for preprocessing the standardized time series data for the LSTM model.
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROCESSED_DATA_PATH, SCALER_PATH, SEQUENCE_LENGTH, TRAIN_SPLIT, VAL_SPLIT

class DataPreprocessor:
    """
    Handles cleaning, scaling, and sequencing of the time series data.
    """
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def run(self):
        """
        Executes the full preprocessing pipeline.
        Returns:
            A tuple containing (train_X, val_X, test_X, scaler).
        """
        print("🚀 Starting Data Preprocessing Pipeline...")
        
        # 1. Load Data
        df = self._load_data(PROCESSED_DATA_PATH)
        if df.empty:
            return None, None, None, None

        # 2. Resample and Impute
        df_resampled = self._resample_and_impute(df)

        # Data Quality Assessment & Visualization
        self._quality_assessment_visualization(df, df_resampled)

        # 3. Scale Data
        scaled_data = self._scale_data(df_resampled)

        # 4. Create Sequences
        sequences = self._create_sequences(scaled_data)

        # 5. Split Data
        train_X, val_X, test_X = self._split_data(sequences)
        
        print("✅ Preprocessing Pipeline Completed.")
        return train_X, val_X, test_X, self.scaler

    def _load_data(self, path):
        """Loads the processed data."""
        if not os.path.exists(path):
            print(f"❌ Error: Processed data not found at '{path}'.")
            print("Please run the Kaggle data loader first.")
            return pd.DataFrame()
        
        print(f"🔄 Loading processed data from '{path}'...")
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        return df

    def _resample_and_impute(self, df):
        """
        Resamples the data to an hourly frequency to identify gaps and
        imputes missing values using linear interpolation.
        """
        print("🕰️ Resampling to hourly frequency and imputing missing values...")
        # Resample to ensure we have a record for every hour in the range
        df_resampled = df.resample('H').asfreq()
        
        missing_before = df_resampled['value'].isnull().sum()
        if missing_before > 0:
            print(f"   -> Found {missing_before} missing hourly records.")
            # Use linear interpolation to fill gaps
            df_resampled['value'] = df_resampled['value'].interpolate(method='linear')
            print("   -> Gaps filled using linear interpolation.")
        else:
            print("   -> No missing hourly records found.")
            
        return df_resampled
    
    def _quality_assessment_visualization(self, original_df, resampled_df):
        """
        Performs data quality checks and creates visualizations.
        """
        print("📊 Performing Data Quality Assessment...")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Data Quality Assessment for NO2 Levels', fontsize=16)

        # 1. Plot original vs. imputed data
        axes[0].plot(original_df.index, original_df['value'], 'o', markersize=2, label='Original Data Points', alpha=0.6)
        axes[0].plot(resampled_df.index, resampled_df['value'], '-', label='Resampled & Imputed Series', color='orange')
        axes[0].set_title('Time Series: Original vs. Resampled & Imputed')
        axes[0].set_ylabel('NO2 (μg/m³)')
        axes[0].legend()

        # 2. Distribution of NO2 values
        sns.histplot(resampled_df['value'], kde=True, ax=axes[1], bins=50)
        axes[1].set_title('Distribution of NO2 Values (after imputation)')
        axes[1].set_xlabel('NO2 (μg/m³)')

        # 3. Monthly Average NO2 Levels
        monthly_avg = resampled_df['value'].resample('M').mean()
        axes[2].plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='--')
        axes[2].set_title('Average Monthly NO2 Levels')
        axes[2].set_ylabel('Average NO2 (μg/m³)')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the plot
        os.makedirs('reports/figures', exist_ok=True)
        plot_path = 'reports/figures/data_quality_assessment.png'
        plt.savefig(plot_path)
        print(f"   -> Quality assessment plot saved to '{plot_path}'")
        plt.close() # Close plot to prevent showing it in non-interactive environments

    def _scale_data(self, df):
        """Scales the 'value' column and saves the scaler."""
        print("📏 Scaling data using MinMaxScaler...")
        values = df['value'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        # Save the scaler
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"   -> Scaler saved to '{SCALER_PATH}'")
        
        return scaled_values

    def _create_sequences(self, data):
        """Creates sequences for the LSTM model."""
        print(f"📜 Creating sequences with length {SEQUENCE_LENGTH}...")
        X = []
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i:(i + SEQUENCE_LENGTH)])
        
        sequences = np.array(X)
        print(f"   -> Generated {sequences.shape[0]} sequences.")
        return sequences

    def _split_data(self, sequences):
        """Splits the data into training, validation, and test sets."""
        print("🔪 Splitting data into train, validation, and test sets...")
        n_sequences = sequences.shape[0]
        train_end = int(n_sequences * TRAIN_SPLIT)
        val_end = train_end + int(n_sequences * VAL_SPLIT)

        train_X = sequences[:train_end]
        val_X = sequences[train_end:val_end]
        test_X = sequences[val_end:]

        print(f"   -> Train set size: {train_X.shape}")
        print(f"   -> Validation set size: {val_X.shape}")
        print(f"   -> Test set size: {test_X.shape}")
        return train_X, val_X, test_X

if __name__ == '__main__':
    print("--- Running Data Preprocessor Standalone Test ---")
    preprocessor = DataPreprocessor()
    train_data, val_data, test_data, saved_scaler = preprocessor.run()

    if train_data is not None:
        print("\n--- Preprocessing Test Summary ---")
        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Scaler saved and can be loaded for inverse transform.")
    else:
        print("\n--- Data preprocessing failed ---")
