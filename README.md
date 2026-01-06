# LUNA25 Docker Submission

Dự án này đóng gói thuật toán I3D để phát hiện nốt phổi (Lung Nodule Detection).

## Cấu trúc dự án
- `Dockerfile`: Cấu hình môi trường chạy.
- `I3D-20251215/`: Chứa weights của model (đã bao gồm).
- `src`: Mã nguồn Python.

## Cách cài đặt và chạy (How to Run)

### 1. Build Docker Image
Mở terminal tại thư mục dự án và chạy:
```bash
docker build -t luna25-i3d .