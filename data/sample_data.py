"""
Sample data generator cho testing khi không có quyền truy cập FIWARE API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_air_quality_data(num_days=30, entities=5):
    """
    Tạo dữ liệu mẫu cho air quality monitoring
    
    Args:
        num_days: Số ngày dữ liệu
        entities: Số lượng sensor entities
    
    Returns:
        DataFrame với format giống FIWARE
    """
    
    # Khởi tạo thời gian
    start_date = datetime.now() - timedelta(days=num_days)
    timestamps = []
    
    # Tạo timestamps mỗi 15 phút
    current_time = start_date
    while current_time < datetime.now():
        timestamps.append(current_time)
        current_time += timedelta(minutes=15)
    
    # Attributes để monitor
    attributes = ['NO2', 'CO', 'PM10', 'PM2.5', 'O3', 'airQualityIndex']
    
    all_data = []
    
    for entity_id in range(1, entities + 1):
        entity_name = f"AirQualityObserved_{entity_id:03d}"
        
        for timestamp in timestamps:
            for attr in attributes:
                # Tạo base value với pattern daily
                hour = timestamp.hour
                day_pattern = np.sin(2 * np.pi * hour / 24) * 0.3
                
                # Base values cho từng pollutant
                if attr == 'NO2':
                    base_value = 40 + day_pattern * 20
                elif attr == 'CO':
                    base_value = 1.5 + day_pattern * 0.8
                elif attr == 'PM10':
                    base_value = 25 + day_pattern * 15
                elif attr == 'PM2.5':
                    base_value = 15 + day_pattern * 10
                elif attr == 'O3':
                    base_value = 80 - day_pattern * 20  # O3 ngược pattern
                else:  # airQualityIndex
                    base_value = 50 + day_pattern * 30
                
                # Thêm noise
                noise = np.random.normal(0, base_value * 0.1)
                value = max(0, base_value + noise)
                
                # Tạo anomalies ngẫu nhiên (5% chance)
                if np.random.random() < 0.05:
                    if attr in ['NO2', 'CO', 'PM10', 'PM2.5']:
                        value *= np.random.uniform(2, 4)  # Spike lên
                    else:
                        value *= np.random.uniform(0.3, 0.7)  # Giảm xuống
                
                record = {
                    'entity_id': entity_name,
                    'attribute': attr,
                    'value': round(value, 2),
                    'timestamp': timestamp.isoformat()
                }
                all_data.append(record)
    
    return pd.DataFrame(all_data)

def save_sample_data():
    """Tạo và lưu sample data"""
    print("Generating sample air quality data...")
    
    # Tạo thư mục nếu chưa có
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate data
    df = generate_sample_air_quality_data(num_days=30, entities=3)
    
    # Save to CSV
    output_path = 'data/raw/air_quality_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Sample data saved to {output_path}")
    print(f"Generated {len(df)} records for {df['entity_id'].nunique()} entities")
    print(f"Attributes: {df['attribute'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    save_sample_data()
