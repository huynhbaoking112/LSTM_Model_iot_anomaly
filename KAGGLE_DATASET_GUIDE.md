# Hướng dẫn tải và sử dụng Kaggle Dataset "Air Quality in Madrid"

Đây là hướng dẫn để tải và thiết lập bộ dữ liệu cần thiết cho dự án.

## 1. Tải Dữ liệu từ Kaggle

-   **Truy cập trang Kaggle**: Mở trình duyệt và truy cập vào đường link sau:
    [https://www.kaggle.com/datasets/decide-soluciones/air-quality-madrid](https://www.kaggle.com/datasets/decide-soluciones/air-quality-madrid)

-   **Đăng nhập và Tải xuống**:
    -   Bạn sẽ cần một tài khoản Kaggle. Nếu chưa có, hãy đăng ký.
    -   Nhấn vào nút "Download" để tải file zip chứa dữ liệu.

## 2. Giải nén và Đặt File vào Project

-   **Giải nén file**: Sau khi tải xong, giải nén file `archive.zip`. Bạn sẽ thấy nhiều file bên trong.

-   **Tìm đúng file**: Chúng ta chỉ cần một file duy nhất là `madrid_2001_2018.csv`.

-   **Đặt file vào thư mục dự án**:
    -   Copy file `madrid_2001_2018.csv`.
    -   Dán file đó vào thư mục `data/raw/` trong project của chúng ta.

-   **Kiểm tra cấu trúc**: Sau khi hoàn tất, cấu trúc thư mục của bạn sẽ trông như thế này:
    ```
    iot/
    ├── data/
    │   ├── raw/
    │   │   └── madrid_2001_2018.csv  <-- File dữ liệu nằm ở đây
    │   └── processed/
    ├── src/
    └── ...
    ```

## 3. Hoàn tất

Sau khi file đã được đặt đúng vị trí, chúng ta đã sẵn sàng để implement module đọc và xử lý dữ liệu này.
