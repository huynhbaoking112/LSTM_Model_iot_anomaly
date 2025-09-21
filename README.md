# Air Quality Anomaly Detection with LSTM

Dự án phát hiện sự kiện bất thường trong dữ liệu cảm biến chất lượng không khí đô thị sử dụng LSTM Autoencoder với dataset từ FIWARE (Santander).

## 🎯 Mục tiêu

- Thu thập dữ liệu chất lượng không khí từ FIWARE API
- Tiền xử lý dữ liệu time series
- Huấn luyện LSTM Autoencoder để phát hiện anomalies
- Đánh giá hiệu suất với precision, recall, F1-score

## 📁 Cấu trúc Project

```
iot/
├── data/
│   ├── raw/                    # Dữ liệu thô từ FIWARE
│   └── processed/              # Dữ liệu đã xử lý
├── models/                     # Models đã train và plots
├── src/
│   ├── __init__.py
│   ├── data_collector.py       # Thu thập dữ liệu từ FIWARE
│   ├── preprocessor.py         # Tiền xử lý dữ liệu
│   ├── lstm_model.py          # LSTM Autoencoder
│   └── evaluator.py           # Đánh giá model
├── config.py                  # Cấu hình hệ thống
├── requirements.txt           # Dependencies
├── main.py                   # Script chính
└── README.md                 # Tài liệu này
```

## 🛠️ Cài đặt

1. **Clone repository và cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Chạy pipeline:**
```bash
python main.py
```

## ⚙️ Cấu hình

Chỉnh sửa `config.py` để thay đổi:

- **FIWARE API endpoint**: Mặc định Santander
- **Model hyperparameters**: Sequence length, epochs, batch size
- **Thresholds**: Ngưỡng phát hiện anomaly
- **File paths**: Đường dẫn lưu trữ

## 🔬 Kiến trúc Model

### LSTM Autoencoder
- **Encoder**: 2 LSTM layers (64 → 32 units)
- **Decoder**: 2 LSTM layers (32 → 64 units)
- **Activation**: ReLU
- **Dropout**: 0.2 để tránh overfitting
- **Loss**: Mean Squared Error (MSE)

### Anomaly Detection
- Tính reconstruction error cho mỗi sequence
- Đặt threshold ở percentile 95% của validation errors
- Sequences có error > threshold được coi là anomalies

## 📊 Đánh giá

Model được đánh giá qua:

- **Unsupervised metrics**: Anomaly rate, reconstruction errors
- **Supervised metrics** (nếu có ground truth): Precision, Recall, F1-score
- **Visualization**: Training curves, error distributions, detected anomalies

## 📈 Kết quả

Sau khi chạy, kết quả sẽ được lưu trong thư mục `models/`:

- `lstm_anomaly_detector.h5`: Model đã train
- `training_history.png`: Biểu đồ loss training
- `evaluation_results.png`: Kết quả đánh giá và anomalies
- `anomalies_plot.png`: Visualization anomalies detected

## 🔧 Tùy chỉnh

### Thay đổi target attribute
Trong `main.py`, thay đổi parameter:
```python
target_attribute='NO2'  # Có thể là 'CO', 'PM10', 'PM2.5', etc.
```

### Điều chỉnh hyperparameters
Trong `config.py`:
```python
SEQUENCE_LENGTH = 24    # Độ dài sequence (hours)
EPOCHS = 50            # Số epochs training
BATCH_SIZE = 32        # Batch size
```

### Fine-tune threshold
```python
ANOMALY_THRESHOLD = 0.1  # Hoặc sử dụng percentile khác
```

## 🚀 Mở rộng

- **Multivariate**: Kết hợp nhiều pollutants cùng lúc
- **Real-time**: Stream processing với Kafka
- **Ensemble**: Kết hợp với các algorithms khác
- **Explainability**: Thêm SHAP values để giải thích anomalies

## 📝 Ghi chú

- Cần internet để truy cập FIWARE API
- Một số API có thể yêu cầu authentication
- Model hiệu quả nhất với dữ liệu có ít nhất 1000+ sequences
- Thời gian training phụ thuộc vào dataset size và hardware

## 🤝 Contribution

Mọi đóng góp và cải thiện đều được chào đón!
