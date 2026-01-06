# 1. Dùng Base Image PyTorch
FROM --platform=linux/amd64 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 2. Tạo User & Cấp quyền
# Tạo user, tạo folder results, và cấp quyền sở hữu cho user này
RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
    && mkdir -p /opt/app/results \
    && chown -R user:user /opt/app

# --- QUAN TRỌNG: CHUYỂN SANG USER MỚI TẠO ---
USER user

# 3. Thiết lập thư mục làm việc
WORKDIR /opt/app

# 4. Cài đặt thư viện
COPY --chown=user:user requirements.txt .
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement requirements.txt

# Cập nhật PATH để hệ thống tìm thấy gunicorn trong thư mục của user
ENV PATH="/home/user/.local/bin:${PATH}"

# 5. Copy Source Code
COPY --chown=user:user . .

RUN mkdir -p resources && mv I3D-20251215 resources/

# 6. Chạy Server
EXPOSE 5000

# Vì đã set ENV PATH ở trên, ta có thể gọi thẳng gunicorn mà không cần đường dẫn dài
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "app:app"]