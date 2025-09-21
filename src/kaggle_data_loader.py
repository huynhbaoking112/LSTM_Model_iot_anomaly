"""
Module to load and preprocess the "Air Quality in Madrid (2001-2018)" dataset from Kaggle.
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
    Handles loading, cleaning, and standardizing the Madrid Air Quality dataset.
    """
    def __init__(self, target_station=28079004):
        """
        Initializes the data loader.
        Args:
            target_station (int): The ID of the station to focus on. 
                                  Default is 28079004 (Plaza de España).
        """
        self.raw_data_path = RAW_DATA_PATH
        self.target_station = target_station
        self.station_col_map = {
            28079004: 'NO_2',  # Plaza de España station has 'NO_2' column for NO2
            # Add other station mappings if needed
        }

    def load_and_process(self):
        """
        Main function to load, clean, and standardize the dataset.
        Returns:
            pd.DataFrame: A standardized DataFrame ready for preprocessing, 
                          or an empty DataFrame if an error occurs.
        """
        if not os.path.exists(self.raw_data_path):
            print(f"❌ Error: Dataset not found at '{self.raw_data_path}'.")
            print("Please follow KAGGLE_DATASET_GUIDE.md to download and place the file.")
            return pd.DataFrame()

        print("🔄 Loading Kaggle dataset...")
        df = self._read_data()

        print(f"🔬 Focusing on station ID: {self.target_station} (Plaza de España)")
        df_station = self._filter_by_station(df)

        if df_station.empty:
            print(f"⚠️ Warning: No data found for station {self.target_station}.")
            return pd.DataFrame()

        print("✨ Standardizing data format...")
        df_standardized = self._standardize_format(df_station)
        
        print("💾 Saving processed data...")
        self._save_processed_data(df_standardized, PROCESSED_DATA_PATH)
        
        print("✅ Kaggle data loading and processing complete.")
        return df_standardized

    def _read_data(self):
        """Reads the raw CSV data."""
        return pd.read_csv(self.raw_data_path)

    def _filter_by_station(self, df):
        """Filters the DataFrame to include only the target station."""
        return df[df['station'] == self.target_station].copy()

    def _standardize_format(self, df_station):
        """
        Transforms the data into the standard format:
        ['timestamp', 'entity_id', 'attribute', 'value']
        """
        # Rename and select columns
        no2_col_name = self.station_col_map.get(self.target_station)
        if no2_col_name not in df_station.columns:
            print(f"❌ Error: NO2 column '{no2_col_name}' not found for station {self.target_station}.")
            return pd.DataFrame()
            
        df_renamed = df_station[['date', no2_col_name]].rename(columns={
            'date': 'timestamp',
            no2_col_name: 'value'
        })

        # Add standard columns
        df_renamed['entity_id'] = f'Madrid-Station-{self.target_station}'
        df_renamed['attribute'] = 'NO2'
        
        # Convert timestamp and handle missing values
        df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])
        df_renamed = df_renamed.dropna(subset=['value'])
        df_renamed = df_renamed.sort_values('timestamp').reset_index(drop=True)

        return df_renamed[['timestamp', 'entity_id', 'attribute', 'value']]
        
    def _save_processed_data(self, df, path):
        """Saves the processed DataFrame to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   -> Saved {len(df)} records to '{path}'")

if __name__ == '__main__':
    print("--- Running Kaggle Data Loader Standalone Test ---")
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
