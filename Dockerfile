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

# 4. 나머지 패키지 설치: requirements.txt 파일 사용 (누락 방지)
# requirements.txt 파일에 명시된 모든 패키지(fastapi, ultralytics, transformers, segment-anything 등)를 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY server.py .

# FastAPI 실행
# 유의: 고객님의 로그에서는 server.py를 사용하므로, 그대로 유지합니다.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}