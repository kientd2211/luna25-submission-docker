# 1. Dùng Base Image PyTorch
FROM --platform=linux/amd64 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN sed -i 's/archive.ubuntu.com/vn.archive.ubuntu.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/vn.archive.ubuntu.com/g' /etc/apt/sources.list

# Cài đặt thư viện (Thêm timeout để không bị treo mãi mãi)
RUN apt-get update --fix-missing \
    && apt-get install -y -o Acquire::Retries=3 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo User & Cấp quyền
RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
    && mkdir -p /opt/app/results \
    && chown -R user:user /opt/app

# --- CHUYỂN SANG USER MỚI ---
USER user

# 3. Thiết lập thư mục làm việc
WORKDIR /opt/app

# 4. Cài đặt thư viện Python
COPY --chown=user:user requirements.txt .
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement requirements.txt

ENV PATH="/home/user/.local/bin:${PATH}"

# 5. Copy Source Code
COPY --chown=user:user . .

RUN mkdir -p resources && mv I3D-20251215 resources/

# 6. Chạy Server
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "app:app"]