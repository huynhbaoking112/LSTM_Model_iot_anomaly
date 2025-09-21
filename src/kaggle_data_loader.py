"""
Module to load and preprocess the "Madrid Polution (2001-2022)" dataset.
"""
import pandas as pd
import numpy as np
import os
import sys

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

class KaggleDataLoader:
    """
    Handles loading, cleaning, and standardizing the Madrid Polution dataset.
    """
    def __init__(self):
        """Initializes the data loader."""
        self.raw_data_path = RAW_DATA_PATH

    def load_and_process(self):
        """
        Main function to load, clean, and standardize the dataset.
        Returns:
            pd.DataFrame: A standardized DataFrame ready for preprocessing, 
                          or an empty DataFrame if an error occurs.
        """
        if not os.path.exists(self.raw_data_path):
            print(f"❌ Error: Dataset not found at '{self.raw_data_path}'.")
            print("Please ensure the file 'MadridPolution2001-2022.csv' is in the 'data/raw/' directory.")
            return pd.DataFrame()

        print("🔄 Loading the new Madrid Polution (2001-2022) dataset...")
        df = self._read_data()

        print("✨ Standardizing data format...")
        df_standardized = self._standardize_format(df)
        
        if df_standardized.empty:
            print("❌ Standardization failed. Please check the input file format.")
            return pd.DataFrame()

        print("💾 Saving processed data...")
        self._save_processed_data(df_standardized, PROCESSED_DATA_PATH)
        
        print("✅ New dataset loading and processing complete.")
        return df_standardized

    def _read_data(self):
        """Reads the raw CSV data."""
        return pd.read_csv(self.raw_data_path)

    def _standardize_format(self, df):
        """
        Transforms the data into the standard format, focusing on NO2.
        ['timestamp', 'entity_id', 'attribute', 'value']
        """
        # Check for required columns
        if 'Time' not in df.columns or 'NO2' not in df.columns:
            print(f"❌ Error: Input CSV must contain 'Time' and 'NO2' columns.")
            return pd.DataFrame()
            
        # Select and rename columns
        df_renamed = df[['Time', 'NO2']].rename(columns={
            'Time': 'timestamp',
            'NO2': 'value'
        })

        # Add standard columns
        df_renamed['entity_id'] = 'Madrid-Polution-TimeSeries'
        df_renamed['attribute'] = 'NO2'
        
        # Convert timestamp and handle missing/invalid values
        df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'], errors='coerce')
        df_renamed['value'] = pd.to_numeric(df_renamed['value'], errors='coerce')
        
        # Drop rows where timestamp or value could not be parsed
        df_renamed.dropna(subset=['timestamp', 'value'], inplace=True)
        
        # Sort and reset index
        df_renamed = df_renamed.sort_values('timestamp').reset_index(drop=True)

        return df_renamed[['timestamp', 'entity_id', 'attribute', 'value']]
        
    def _save_processed_data(self, df, path):
        """Saves the processed DataFrame to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   -> Saved {len(df)} records to '{path}'")

if __name__ == '__main__':
    print("--- Running Kaggle Data Loader Standalone Test for new dataset ---")
    loader = KaggleDataLoader()
    processed_df = loader.load_and_process()

    if not processed_df.empty:
        print("\n--- Processed Data Sample ---")
        print(processed_df.head())
        print("\n--- Data Info ---")
        processed_df.info()
        print("\n--- Basic Stats ---")
        print(processed_df['value'].describe())
    else:
        print("\n--- Data processing failed ---")
