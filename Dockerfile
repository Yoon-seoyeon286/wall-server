FROM python:3.10-slim

WORKDIR /app

# 1. 필수 시스템 패키지 설치
# OpenGL 관련 패키지 (libgl1, libglib2.0-0)는 OpenCV-Python의 필수 의존성입니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# 2. pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Torch CPU 버전 설치
# PyTorch와 그 의존성(torchvision, torchaudio)을 먼저 설치합니다.
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. 나머지 패키지 설치 (NumPy 및 Ultralytics 포함)
# NumPy를 명시적으로 설치하여 Ultralytics가 의존성을 올바르게 찾도록 합니다.
# 안정적인 버전 1.26.4를 지정합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    opencv-python-headless \
    psutil \
    numpy==1.26.4 \
    ultralytics==8.3.20

# 앱 코드 복사
COPY server.py .

# FastAPI 실행
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}