# Kế hoạch Triển khai Hệ thống Cảnh báo Bất thường

Đây là kế hoạch chi tiết từng bước để tích hợp tính năng gửi cảnh báo từ `predictor` đến một server API.

## Giai đoạn 1: Chuẩn bị Phía Client (`predictor`)

-   [x] **Cập nhật Cấu hình (`config.py`)**
    -   [x] Thêm biến `ALERTING_ENABLED` để có thể bật/tắt tính năng gửi cảnh báo.
    -   [x] Thêm biến `ALERTING_API_ENDPOINT` và đặt giá trị mặc định là `http://127.0.0.1:8000/api/alerts`.

-   [x] **Thiết kế Cấu trúc Gói tin Cảnh báo (Alert Payload)**
    -   [x] Thống nhất các trường dữ liệu cần thiết sẽ được gửi đi trong mỗi cảnh báo.
    -   [x] Đảm bảo payload chứa đủ thông tin chi tiết để frontend có thể vẽ lại biểu đồ (timestamps, giá trị gốc, giá trị tái tạo).
    -   [x] Các trường chính: `sensor_id`, `alert_timestamp`, `anomaly_score`, `threshold`, và `sequence_data`.

-   [x] **Nâng cấp `src/predictor.py`**
    -   [x] Tạo một hàm mới `send_alert_to_server(payload)` để xử lý việc gửi HTTP POST request.
    -   [x] Tích hợp thư viện `requests` để thực hiện gửi request.
    -   [x] Trong hàm `predict_anomalies`, sau khi vòng lặp phát hiện các chuỗi bất thường, hãy gọi hàm `send_alert_to_server` cho mỗi bất thường.
    -   [x] Thêm cơ chế xử lý lỗi (ví dụ: `try...except`) để `predictor` không bị dừng đột ngột nếu server không phản hồi.
    -   [x] Chỉ gửi request khi `ALERTING_ENABLED` trong config là `True`.

## Giai đoạn 2: Xây dựng Server Cảnh báo (FastAPI)

-   [x] **Thiết lập Cấu trúc Thư mục cho Server**
    -   [x] Tạo một thư mục mới ở gốc dự án tên là `alert_server/`.
    -   [x] Trong `alert_server/`, tạo các file `main.py`, `database.py`, `models.py`, và `requirements.txt`.

-   [x] **Định nghĩa Mô hình Dữ liệu (`models.py`)**
    -   [x] Sử dụng Pydantic để tạo các class tương ứng với cấu trúc JSON của Alert Payload.
    -   [x] Việc này giúp FastAPI tự động validate dữ liệu nhận được, đảm bảo tính toàn vẹn.

-   [x] **Thiết lập Kết nối Cơ sở dữ liệu (`database.py`)**
    -   [x] Viết code để kết nối đến MongoDB Atlas bằng chuỗi kết nối đã cung cấp.
    -   [x] Tạo hàm `save_alert(alert_data)` để nhận dữ liệu từ API và lưu vào collection `alerts`.
    -   [x] Đảm bảo xử lý các lỗi kết nối đến DB một cách an toàn.
    -   [x] Lấy chuỗi kết nối từ biến môi trường để tăng tính bảo mật (thay vì hardcode).

-   [x] **Xây dựng API Endpoint (`main.py`)**
    -   [x] Khởi tạo một ứng dụng FastAPI.
    -   [x] Tạo một endpoint `POST /api/alerts`.
    -   [x] Endpoint này sẽ nhận payload đã được validate bởi Pydantic model.
    -   [x] Gọi hàm `save_alert` từ `database.py` để lưu cảnh báo.
    -   [x] Trả về một JSON response cho client để xác nhận đã nhận được cảnh báo.

-   [x] **Ghi vào `requirements.txt` cho Server**
    -   [x] Thêm các thư viện cần thiết: `fastapi`, `uvicorn`, `pydantic`, `pymongo[srv]`.

## Giai đoạn 3: Kiểm thử Tích hợp (Integration Testing)

-   [ ] **Kiểm thử trên Local**
    -   [ ] Chạy server FastAPI trên local: `uvicorn alert_server.main:app --reload`.
    -   [ ] Chạy `predictor.py` với một file test bất thường.
    -   [ ] Kiểm tra log của server FastAPI để xem có nhận được request không.
    -   [ ] Đăng nhập vào MongoDB Atlas và kiểm tra xem document cảnh báo mới đã được tạo trong collection `alerts` chưa.

-   [ ] **Kiểm thử Xử lý Lỗi**
    -   [ ] Tắt server FastAPI và chạy lại `predictor.py`.
    -   [ ] Xác nhận rằng `predictor` vẫn chạy đến cùng mà không bị crash, chỉ in ra lỗi không thể kết nối.

## Giai đoạn 4: Deployment Lên VPS

-   [ ] **Chuẩn bị Môi trường VPS**
    -   [ ] Cài đặt Python, pip, và Nginx trên server.
    -   [ ] Sao chép thư mục `alert_server/` lên VPS.
    -   [ ] Cài đặt các thư viện từ `requirements.txt` của server.

-   [ ] **Cấu hình Web Server Production**
    -   [ ] Sử dụng Gunicorn hoặc Uvicorn để chạy ứng dụng FastAPI một cách ổn định.
    -   [ ] Viết một file service `systemd` để quản lý tiến trình của server (tự động khởi động, restart khi có lỗi).

-   [ ] **Cấu hình Reverse Proxy (Nginx)**
    -   [ ] Cấu hình Nginx để chuyển tiếp các request từ port 80 (HTTP) đến ứng dụng FastAPI đang chạy trên port 8000.
    -   [ ] (Tùy chọn) Cấu hình SSL với Let's Encrypt để bật HTTPS.

-   [ ] **Kiểm thử Cuối cùng**
    -   [ ] Cập nhật `ALERTING_API_ENDPOINT` trong `config.py` thành địa chỉ IP hoặc tên miền của VPS.
    -   [ ] Chạy `predictor.py` từ máy local và kiểm tra xem cảnh báo có được lưu vào MongoDB trên cloud không.

---
**Hoàn thành các bước trên sẽ tạo ra một hệ thống cảnh báo end-to-end hoàn chỉnh và mạnh mẽ.**
