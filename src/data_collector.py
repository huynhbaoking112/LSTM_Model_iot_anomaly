import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
from config import *

class FIWAREDataCollector:
    def __init__(self):
        self.base_url = FIWARE_BASE_URL
        self.entity_type = FIWARE_ENTITY_TYPE
        
    def get_entities(self, limit=100):
        """Lấy danh sách entities từ FIWARE"""
        url = f"{self.base_url}/v2/entities"
        headers = {'Accept': 'application/json'}
        params = {
            'type': self.entity_type,
            'limit': limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching entities: {e}")
            return []
    
    def get_entity_data(self, entity_id, start_date=None, end_date=None):
        """Lấy dữ liệu lịch sử của một entity"""
        url = f"{self.base_url}/v2/entities/{entity_id}/attrs"
        headers = {'Accept': 'application/json'}
        
        if start_date and end_date:
            params = {
                'dateFrom': start_date.isoformat(),
                'dateTo': end_date.isoformat()
            }
            response = requests.get(url, headers=headers, params=params)
        else:
            response = requests.get(url, headers=headers)
            
        response.raise_for_status()
        return response.json()
    
    def collect_historical_data(self, days=30):
        """Thu thập dữ liệu lịch sử"""
        entities = self.get_entities()
        all_data = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for entity in entities:
            entity_id = entity['id']
            try:
                data = self.get_entity_data(entity_id, start_date, end_date)
                if data:
                    # Flatten nested data
                    flat_data = self._flatten_entity_data(data, entity_id)
                    all_data.extend(flat_data)
            except Exception as e:
                print(f"Error collecting data for {entity_id}: {e}")
                
        return pd.DataFrame(all_data)
    
    def _flatten_entity_data(self, data, entity_id):
        """Chuyển đổi dữ liệu nested thành flat format"""
        records = []
        
        # Get all attributes except metadata
        attrs = {k: v for k, v in data.items() if k not in ['id', 'type']}
        
        for attr_name, attr_data in attrs.items():
            if 'value' in attr_data:
                record = {
                    'entity_id': entity_id,
                    'attribute': attr_name,
                    'value': attr_data['value'],
                    'timestamp': attr_data.get('metadata', {}).get('timestamp', {}).get('value')
                }
                records.append(record)
                
        return records
    
    def save_to_csv(self, df, filepath):
        """Lưu dữ liệu vào CSV"""
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

# Usage example
if __name__ == "__main__":
    collector = FIWAREDataCollector()
    
    # Collect 30 days of historical data
    print("Collecting historical data...")
    df = collector.collect_historical_data(days=30)
    
    if not df.empty:
        collector.save_to_csv(df, RAW_DATA_PATH)
        print(f"Collected {len(df)} records")
    else:
        print("No data collected")
