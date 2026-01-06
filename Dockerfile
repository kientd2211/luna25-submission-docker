# Sử dụng base image PyTorch
FROM --platform=linux/amd64 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Tránh buffer log để debug dễ hơn
ENV PYTHONUNBUFFERED=1

# Tạo user để không chạy dưới quyền root (yêu cầu bảo mật của Grand Challenge)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# --- 1. Cài đặt thư viện trước (Tận dụng Docker Cache) ---
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# --- 2. Copy Source Code ---
# Copy các file code Python quan trọng
COPY --chown=user:user inference.py processor.py dataloader.py experiment_config.py /opt/app/
# Copy thư mục chứa kiến trúc mạng (Network Architecture)
COPY --chown=user:user models /opt/app/models

# --- 3. Copy Model Weights (Quan trọng) ---
# Cách 1: Nếu bạn muốn chạy model 2D (LUNA25-baseline-2D-20250225)
# COPY --chown=user:user results /opt/app/resources

# Cách 2: Nếu bạn muốn chạy model I3D (I3D-20251215)
# Chúng ta tạo thư mục resources trước, sau đó copy folder I3D vào trong đó
RUN mkdir -p /opt/app/resources
COPY --chown=user:user I3D-20251215 /opt/app/resources/I3D-20251215

# Lưu ý: Bạn chỉ nên chọn 1 trong 2 cách copy trên tùy vào việc bạn muốn đóng gói model nào
# Hiện tại tôi đang để Cách 2 (I3D) vì bạn có nhắc đến I3D trước đó.

# Copy file license hoặc các file phụ trợ khác nếu cần
# COPY --chown=user:user test /opt/app/test

ENTRYPOINT ["python", "inference.py"]