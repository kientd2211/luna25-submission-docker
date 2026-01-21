# LUNA25 - Hệ Thống Chẩn Đoán Ung Thư Phổi (Patient-Level)

Hệ thống dự đoán khả năng ác tính của nốt phổi (Lung Nodule Malignancy Prediction) ở mức độ ca bệnh. Hệ thống tích hợp quy trình xử lý ảnh y tế hiện đại:
1.  **Segmentation:** Sử dụng mô hình **MedSAM** để tách nốt phổi khỏi nền.
2.  **Classification:** Sử dụng mô hình **I3D (Inflated 3D ConvNet)** để phân loại ác tính/lành tính.
3.  **Deployment:** Đóng gói hoàn chỉnh bằng **Docker**.

---

## 1. Cấu Trúc Dự Án

```text
luna25-clean/
├── backend/                  # Mã nguồn Server (Flask API)
│   ├── resources/            # Chứa tài nguyên model (MedSAM)
│   ├── I3D-20251215/         # Chứa trọng số model phân loại (I3D)
│   ├── app.py                # API Entrypoint
│   ├── inference.py          # Logic xử lý chính (MedSAM + I3D pipeline)
│   ├── processor.py          # Class xử lý model gốc
│   ├── dataloader.py         # Xử lý dữ liệu ảnh CT
│   └── Dockerfile            # Cấu hình Docker Image
├── tools/                    # Các công cụ hỗ trợ
│   ├── convert_all_series.py # Script chuyển đổi DICOM -> MHA
│   └── get_valid_coords.py   # Tìm tọa độ tâm ảnh để test
├── output_mha/               # Thư mục chứa file ảnh sau khi convert
├── test_patient.py           
├── run_submission.py           
├── docker-compose.yml        # Cấu hình chạy Container
└── requirements.txt          # Các thư viện Python cần thiết
```

## 2. Yêu Cầu Cài Đặt

### Tải Model Weights
Hệ thống sẽ không hoạt động nếu thiếu các file này. Vui lòng tải và đặt chính xác vào thư mục chỉ định:

1.  **MedSAM Checkpoint (`medsam_vit_b.pth`)**
    * **Tải tại:** [MedSAM Google Drive](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) hoặc [Repo chính thức](https://github.com/bowang-lab/MedSAM).
    * **Đổi tên và đặt vào:** `backend/resources/MedSAM/medsam_vit_b.pth`
    * *Lưu ý: Nếu thiếu file này, hệ thống sẽ tự động bỏ qua bước Segmentation và chỉ chạy phân loại thô.*

2.  **I3D Model (`best_metric_model.pth`)**
    * **(Nếu chưa có) Tải tại:** ([Link Drive của nhóm đồ án](https://drive.google.com/drive/folders/12d3aEn-p0TqrJUuMVeCgUJej-_ylHHFJ?usp=drive_link)).
    * **Đặt vào:** `backend/I3D-20251215/best_metric_model.pth`

## 3. Hướng Dẫn Chạy (Quick Start)

### Bước 1: Khởi động Server
Mở terminal tại thư mục gốc của dự án và chạy lệnh:

```bash
docker-compose up --build
```

### Bước 2: Chạy Demo Chấm Điểm (Dành cho Giảng Viên)
Script `run_submission.py` để chạy trực tiếp trên **Folder DICOM**.

1.  Mở file `run_submission.py`.
2.  Sửa đường dẫn tại biến `DICOM_FOLDER` trỏ tới thư mục chứa ảnh CT.
    ```python
    DICOM_FOLDER = r"C:\Data\Test_Set\Patient_01"
    ```
3.  Chạy lệnh:
    ```bash
    python run_submission.py
    ```

**Kết quả output mẫu:**
```json
{
  "seriesInstanceUID": "Patient_01",
  "probability": 0.8523,
  "predictionLabel": 1,
  "processingTimeMs": 8500,
  "CoordX": 0.0,
  "CoordY": 0.0,
  "CoordZ": -100.0
}
```

### Bước 3: Chạy Test với File MHA
Nếu đã có sẵn file ảnh định dạng `.mha` (ví dụ file trong thư mục `output_mha/` sau khi convert), có thể kiểm thử trực tiếp API bằng script `test_patient.py`.

1.  Mở file `test_patient.py`.
2.  Kiểm tra/Sửa đường dẫn file ảnh muốn test (thường nằm ở đầu file hoặc trong hàm `main`):
    ```python
    file_path = "C:\Data\Test_Set\Patient_01\3-1.25mm_NHU_MO_PHOI.mha"
    ```
3.  Chạy lệnh:
    ```bash
    python test_patient.py
    ```

**Kết quả output mẫu:**
```json
[
  {
    "seriesInstanceUID": "1",
    "probability": 0.5057,
    "predictionLabel": 1,
    "processingTimeMs": 8033,
    "CoordX": 154.42,
    "CoordY": 105.35,
    "CoordZ": -163.54
  }
]
```

## 4. Thông Tin Kỹ Thuật

### Quy trình xử lý

Hệ thống xử lý dữ liệu theo luồng tuần tự để đảm bảo độ chính xác cao nhất về mặt không gian:

1.  **Input Handling:**
    * Server nhận file ảnh CT (`.mha`) chứa dữ liệu khối 3D.
    * Nhận danh sách tọa độ nốt phổi (Candidates) dưới dạng World Coordinates (mm) từ hệ thống CAD/Bác sĩ.

2.  **Coordinate Mapping:**
    * Sử dụng thư viện **SimpleITK** để chuyển đổi trực tiếp từ *Physical Point (mm)* sang *Voxel Index (x, y, z)* trên CPU.
    * **Kỹ thuật Bypass Transform:** Dữ liệu đưa vào mô hình Deep Learning được chuẩn hóa hướng trục để tránh các lỗi lệch trục không gian thường gặp khi xử lý ảnh y tế.

3.  **Segmentation (MedSAM Integration):**
    * Tại lát cắt 2D chứa tâm nốt phổi, hệ thống tự động tạo một **Bounding Box 40mm**.
    * Gửi Prompt vào mô hình **MedSAM (ViT-B)** để tạo mặt nạ tách nốt phổi.
    * Áp dụng mask lên ảnh gốc để loại bỏ nền nhiễu (mạch máu, thành phổi...) trước khi phân loại.

4.  **Classification :**
    * Khối ảnh 3D đã được crop và segment được đưa vào mô hình **I3D**.
    * Kết quả trả về là xác suất ác tính từ 0.0 đến 1.0.

### API Documentation

* **Endpoint:** `POST /api/v1/predict/patient`
* **Content-Type:** `multipart/form-data`

| Tham số | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `file` | File | File ảnh CT định dạng `.mha`. |
| `seriesInstanceUID` | String | Mã định danh của chuỗi ảnh. |
| `candidates` | JSON String | Danh sách tọa độ nốt phổi. |

**Cấu trúc JSON `candidates`:**
```json
[
  {
    "seriesInstanceUID": "1.2.3...", 
    "CoordX": -128.5, 
    "CoordY": 40.2, 
    "CoordZ": -100.0
  }
]
```