FROM python:3.10-slim

WORKDIR /app

# 1. 필수 시스템 패키지 설치
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

# 4. 나머지 패키지 설치 (안정성 확보를 위해 명시적으로 설치 순서 조정)
COPY requirements.txt .

RUN pip install --no-cache-dir \
    # 웹 서버 및 기본 라이브러리
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    Pillow==10.1.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.26.4 \
    psutil==5.9.6

RUN pip install --no-cache-dir \
    # AI/ML 라이브러리
    ultralytics==8.3.20 \
    transformers==4.36.0 \
    segment-anything==1.0


# 앱 코드 복사
COPY server.py .

# FastAPI 실행
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}