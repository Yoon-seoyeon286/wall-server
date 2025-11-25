FROM python:3.10-slim

WORKDIR /app

# 필수 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Numpy 먼저 설치
RUN pip install --no-cache-dir "numpy<2"

# Torch CPU 버전 설치
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    opencv-python-headless \
    psutil \
    ultralytics==8.3.20

# 앱 코드 복사
COPY server.py .

# FastAPI 실행
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}