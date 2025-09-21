from src.data_collector import FIWAREDataCollector
from src.preprocessor import DataPreprocessor
from src.lstm_model import LSTMAutoencoder, plot_training_history
from src.evaluator import AnomalyEvaluator, print_evaluation_summary
from config import *
import os

def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = ['data/raw', 'data/processed', 'models']
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def main():
    """Main pipeline"""
    print("Starting Air Quality Anomaly Detection Pipeline...")
    
    # Tạo thư mục
    create_directories()
    
    # 1. Thu thập dữ liệu
    print("\n1. Collecting data from FIWARE...")
    collector = FIWAREDataCollector()
    
    # Kiểm tra xem đã có dữ liệu chưa
    if not os.path.exists(RAW_DATA_PATH):
        print("Fetching fresh data...")
        df = collector.collect_historical_data(days=30)
        collector.save_to_csv(df, RAW_DATA_PATH)
    else:
        print("Using existing data...")
    
    # 2. Tiền xử lý dữ liệu
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    train_data, val_data, test_data, scaler = preprocessor.prepare_data_for_training(
        RAW_DATA_PATH, target_attribute='NO2'
    )
    
    # 3. Xây dựng và huấn luyện model
    print("\n3. Training LSTM Autoencoder...")
    model = LSTMAutoencoder(sequence_length=SEQUENCE_LENGTH, n_features=1)
    model.build_model()
    
    # Huấn luyện
    history = model.train(train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Lưu model
    model.save_model(MODEL_PATH)
    
    # Đặt threshold
    threshold = model.set_threshold(val_data)
    
    # 4. Đánh giá trên test data
    print("\n4. Evaluating model...")
    evaluator = AnomalyEvaluator(model, threshold)
    results, anomalies, errors = evaluator.evaluate(test_data)
    
    # In kết quả
    print_evaluation_summary(results)
    
    # Vẽ biểu đồ
    evaluator.plot_evaluation_results(results, errors, anomalies)
    plot_training_history(history)
    
    print("\nPipeline completed successfully!")
    print("Check the 'models/' directory for saved model and plots.")

if __name__ == "__main__":
    main()
