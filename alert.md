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

-   [x] **Kiểm thử trên Local**
    -   [x] Chạy server FastAPI trên local: `uvicorn alert_server.main:app --reload`.
    -   [x] Chạy `predictor.py` với một file test bất thường.
    -   [x] Kiểm tra log của server FastAPI để xem có nhận được request không.
    -   [x] Đăng nhập vào MongoDB Atlas và kiểm tra xem document cảnh báo mới đã được tạo trong collection `alerts` chưa.

-   [x] **Kiểm thử Xử lý Lỗi**
    -   [x] Tắt server FastAPI và chạy lại `predictor.py`.
    -   [x] Xác nhận rằng `predictor` vẫn chạy đến cùng mà không bị crash, chỉ in ra lỗi không thể kết nối.

## Giai đoạn 4: Xây dựng Giao diện Giám sát (SSR)

-   [ ] **Cập nhật Dependencies & Cấu hình**
    -   [ ] Thêm `Jinja2` vào `alert_server/requirements.txt`.
    -   [ ] Cấu hình FastAPI để sử dụng Jinja2 templates.
    -   [ ] Tạo thư mục `templates/` và `static/` trong `alert_server/`.

-   [ ] **Thiết kế Trang Dashboard**
    -   [ ] Tạo tệp `templates/index.html`.
    -   [ ] Viết mã HTML để hiển thị danh sách các cảnh báo (ID, thời gian, điểm bất thường).
    -   [ ] Thêm một chút CSS trong `static/styles.css` để giao diện trông gọn gàng hơn.

-   [ ] **Nâng cấp Server**
    -   [ ] Viết một hàm mới trong `database.py` để truy vấn tất cả các cảnh báo từ MongoDB.
    -   [ ] Tạo một endpoint mới `GET /dashboard` trong `main.py`.
    -   [ ] Endpoint này sẽ lấy dữ liệu từ database và render trang `index.html` với dữ liệu đó.

-   [x] **Kiểm thử Giao diện**
    -   [x] Chạy lại server.
    -   [x] Mở trình duyệt và truy cập `http://127.0.0.1:8000/dashboard`.
    -   [x] Xác nhận rằng các cảnh báo đã được gửi trước đó hiển thị chính xác trên trang.

-   [x] **Nâng cấp Giao diện Chi tiết (với Biểu đồ)**
    -   [x] Thêm thư viện `Chart.js` vào thư mục `static/`.
    -   [x] Viết hàm `get_alert_by_id` trong `database.py`.
    -   [x] Tạo template `alert_detail.html` với thẻ `<canvas>` cho biểu đồ.
    -   [x] Tạo endpoint `GET /dashboard/alert/{alert_id}` trong `main.py` để render trang chi tiết.
    -   [x] Cập nhật link trong `index.html` để trỏ đến trang chi tiết.
    -   [x] Viết mã JavaScript trong `alert_detail.html` để vẽ biểu đồ từ dữ liệu được truyền vào.

---
**Hoàn thành các bước trên sẽ tạo ra một hệ thống cảnh báo end-to-end hoàn chỉnh và mạnh mẽ.**
