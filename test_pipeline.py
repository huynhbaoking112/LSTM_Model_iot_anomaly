"""
Test script để kiểm tra pipeline hoạt động với sample data
"""
import os
import sys
from data.sample_data import save_sample_data
from main import main

def test_with_sample_data():
    """Test pipeline với sample data"""
    print("🧪 Testing Air Quality Anomaly Detection Pipeline")
    print("=" * 50)
    
    # 1. Tạo sample data
    print("\n1. Generating sample data...")
    save_sample_data()
    
    # 2. Chạy pipeline
    print("\n2. Running main pipeline...")
    try:
        main()
        print("\n✅ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return False
    
    # 3. Kiểm tra outputs
    print("\n3. Checking outputs...")
    expected_files = [
        'data/raw/air_quality_data.csv',
        'models/lstm_anomaly_detector.h5',
        'models/best_lstm_model.h5'
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing!")
    
    print("\n🎉 Test completed!")
    return True

if __name__ == "__main__":
    test_with_sample_data()
